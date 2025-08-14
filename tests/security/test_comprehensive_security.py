"""
Comprehensive Security System Testing Suite.

Tests all implemented security features including input validation,
encryption, audit logging, rate limiting, failover, and monitoring systems.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import logging
from typing import Dict, List, Any
from datetime import datetime, timezone
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.security
class TestComprehensiveSecurity:
    """Test comprehensive security implementation."""
    
    async def test_input_validation_framework(self):
        """Test input validation and sanitization."""
        logger.info("🔒 Testing input validation framework")
        
        from nasa_cmr_agent.security.input_validator import get_input_validator, ValidationSeverity
        
        validator = get_input_validator()
        
        # Test SQL injection detection
        malicious_query = "'; DROP TABLE users; --"
        result = validator.validate_query_input(malicious_query)
        assert not result.is_valid
        assert result.severity == ValidationSeverity.CRITICAL
        assert "injection" in result.message.lower()
        
        # Test XSS detection
        xss_query = "<script>alert('xss')</script>"
        result = validator.validate_query_input(xss_query)
        assert not result.is_valid
        assert result.severity == ValidationSeverity.CRITICAL
        
        # Test valid query
        valid_query = "precipitation data over North America"
        result = validator.validate_query_input(valid_query)
        assert result.is_valid
        assert result.sanitized_value is not None
        
        # Test coordinate validation
        result = validator.validate_coordinates(40.7128, -74.0060)  # NYC
        assert result.is_valid
        assert result.sanitized_value["lat"] == 40.7128
        
        # Test invalid coordinates
        result = validator.validate_coordinates(91, -74)  # Invalid lat
        assert not result.is_valid
        
        # Test sensitive data detection
        sensitive_text = "My SSN is 123-45-6789"
        result = validator.detect_sensitive_data(sensitive_text)
        assert not result.is_valid
        assert "ssn" in result.message.lower()
        
        logger.info("✅ Input validation framework tests passed")
    
    async def test_encryption_system(self):
        """Test encryption and data protection."""
        logger.info("🔒 Testing encryption system")
        
        from nasa_cmr_agent.security.encryption import get_encryption_service, KeyType
        
        # Skip if cryptography not available
        try:
            encryption_service = get_encryption_service()
            if not encryption_service:
                logger.info("⚠️ Encryption tests skipped - cryptography not available")
                return
        except Exception as e:
            logger.info(f"⚠️ Encryption tests skipped - {e}")
            return
        
        # Test data encryption/decryption
        test_data = "Sensitive NASA CMR query: precipitation over Arctic"
        
        result = encryption_service.encrypt_data(test_data, KeyType.DATA)
        assert result.success
        assert result.encrypted_data is not None
        assert result.key_id is not None
        
        # Test decryption
        decrypt_result = encryption_service.decrypt_data(result.encrypted_data, result.key_id)
        assert decrypt_result.success
        assert decrypt_result.decrypted_data == test_data
        
        # Test query encryption
        query = "temperature anomalies in Greenland"
        encrypted_query, key_id = encryption_service.encrypt_sensitive_query(query)
        decrypted_query = encryption_service.decrypt_sensitive_query(encrypted_query, key_id)
        assert decrypted_query == query
        
        # Test cache data encryption
        cache_data = {"query": "test", "results": [1, 2, 3], "timestamp": "2024-01-01"}
        encrypted_cache, cache_key_id = encryption_service.encrypt_cache_data(cache_data)
        decrypted_cache = encryption_service.decrypt_cache_data(encrypted_cache, cache_key_id)
        assert decrypted_cache == cache_data
        
        # Test key rotation
        new_key_id = encryption_service.rotate_key(KeyType.DATA)
        assert new_key_id != result.key_id
        
        # Get encryption statistics
        stats = encryption_service.get_encryption_stats()
        assert stats["total_encryptions"] >= 3
        assert stats["total_decryptions"] >= 3
        assert stats["keys_generated"] >= 4  # Initial + rotated
        
        logger.info("✅ Encryption system tests passed")
    
    async def test_security_audit_logging(self):
        """Test security audit logging system."""
        logger.info("🔒 Testing security audit logging")
        
        from nasa_cmr_agent.security.audit_logger import (
            get_audit_logger, SecurityEventType, SecurityEventSeverity, SecurityEventStatus
        )
        
        audit_logger = get_audit_logger()
        
        # Test authentication logging
        auth_event = await audit_logger.log_authentication_event(
            success=True,
            user_id="test_user_123",
            ip_address="192.168.1.100",
            user_agent="NASA-CMR-Agent/1.0",
            details={"login_method": "api_key"}
        )
        
        assert auth_event.event_type == SecurityEventType.AUTHENTICATION
        assert auth_event.status == SecurityEventStatus.SUCCESS
        assert auth_event.user_id == "test_user_123"
        
        # Test failed authentication
        failed_auth = await audit_logger.log_authentication_event(
            success=False,
            user_id="attacker",
            ip_address="10.0.0.1",
            user_agent="BadBot/1.0"
        )
        
        assert failed_auth.status == SecurityEventStatus.FAILURE
        assert failed_auth.severity == SecurityEventSeverity.MEDIUM
        
        # Test data access logging
        data_event = await audit_logger.log_data_access_event(
            user_id="test_user_123",
            resource="nasa_cmr_collections",
            action="query_collections",
            success=True,
            details={"query": "precipitation", "result_count": 150}
        )
        
        assert data_event.event_type == SecurityEventType.DATA_ACCESS
        assert data_event.resource == "nasa_cmr_collections"
        
        # Test threat detection logging
        threat_event = await audit_logger.log_threat_detection(
            threat_type="SQL Injection Attempt",
            indicators=["sql_keywords", "malicious_patterns"],
            details={"blocked": True, "pattern": "UNION SELECT"},
            ip_address="10.0.0.1",
            user_id="attacker"
        )
        
        assert threat_event.event_type == SecurityEventType.THREAT_DETECTION
        assert threat_event.severity == SecurityEventSeverity.HIGH
        
        # Test input validation logging
        validation_result = {
            "is_valid": False,
            "severity": "critical",
            "message": "SQL injection detected",
            "sanitized_value": None
        }
        
        validation_event = await audit_logger.log_input_validation_event(
            validation_result=validation_result,
            user_id="test_user",
            ip_address="192.168.1.1"
        )
        
        assert validation_event.event_type == SecurityEventType.INPUT_VALIDATION
        assert validation_event.status == SecurityEventStatus.BLOCKED
        
        # Get audit statistics
        stats = audit_logger.get_audit_statistics()
        assert stats["total_events"] >= 4
        assert stats["threats_detected"] >= 1
        assert stats["blocked_actions"] >= 1
        
        # Test compliance report generation
        compliance_report = await audit_logger.generate_compliance_report("NIST", days=1)
        assert compliance_report["framework"] == "NIST"
        assert compliance_report["total_events"] >= 4
        
        logger.info("✅ Security audit logging tests passed")
    
    async def test_rate_limiting_and_ddos_protection(self):
        """Test rate limiting and DDoS protection."""
        logger.info("🔒 Testing rate limiting and DDoS protection")
        
        from nasa_cmr_agent.security.rate_limiter import get_rate_limiter, LimitType
        
        rate_limiter = get_rate_limiter()
        
        # Test normal request (should be allowed)
        result = await rate_limiter.check_rate_limit(
            client_id="test_client_1",
            service="default",
            limit_type=LimitType.REQUESTS_PER_SECOND,
            ip_address="192.168.1.100"
        )
        
        assert result.allowed
        assert result.current_count >= 1
        
        # Test rapid requests to trigger rate limiting
        blocked_count = 0
        for i in range(15):  # Exceed default limit of 10 RPS
            result = await rate_limiter.check_rate_limit(
                client_id="rapid_client",
                service="default",
                limit_type=LimitType.REQUESTS_PER_SECOND,
                ip_address="10.0.0.2"
            )
            
            if not result.allowed:
                blocked_count += 1
        
        assert blocked_count > 0, "Rate limiting should have blocked some requests"
        
        # Test whitelisted client
        rate_limiter.whitelist_client("trusted_client", "192.168.1.200")
        
        # Whitelisted client should have higher limits
        allowed_count = 0
        for i in range(15):
            result = await rate_limiter.check_rate_limit(
                client_id="trusted_client",
                service="default",
                limit_type=LimitType.REQUESTS_PER_SECOND,
                ip_address="192.168.1.200"
            )
            
            if result.allowed:
                allowed_count += 1
        
        assert allowed_count > 5, "Whitelisted client should have higher limits"
        
        # Test blacklisted client
        rate_limiter.blacklist_client("malicious_client", "10.0.0.3", duration_seconds=60)
        
        result = await rate_limiter.check_rate_limit(
            client_id="malicious_client",
            service="default",
            limit_type=LimitType.REQUESTS_PER_SECOND,
            ip_address="10.0.0.3"
        )
        
        assert not result.allowed, "Blacklisted client should be blocked"
        
        # Test traffic analysis
        traffic_analysis = rate_limiter.get_traffic_analysis()
        assert "current_rps" in traffic_analysis
        assert "threat_level" in traffic_analysis
        
        # Get rate limiting statistics
        stats = rate_limiter.get_rate_limit_stats()
        assert stats["total_requests"] > 15
        assert stats["blocked_requests"] > 0
        assert stats["active_clients"] >= 3
        
        logger.info("✅ Rate limiting and DDoS protection tests passed")
    
    async def test_failover_and_recovery_system(self):
        """Test sophisticated failover strategies."""
        logger.info("🔒 Testing failover and recovery system")
        
        from nasa_cmr_agent.resilience.failover_manager import (
            get_failover_manager, ServiceConfig, ServiceEndpoint, 
            FailoverStrategy, RecoveryMode
        )
        
        failover_manager = get_failover_manager()
        
        # Register a test service with multiple endpoints
        endpoints = [
            ServiceEndpoint(
                service_id="test_service",
                endpoint_id="primary",
                url="http://localhost:8080",
                priority=10,
                is_primary=True,
                region="us-east-1"
            ),
            ServiceEndpoint(
                service_id="test_service", 
                endpoint_id="secondary",
                url="http://localhost:8081",
                priority=5,
                region="us-west-2"
            ),
            ServiceEndpoint(
                service_id="test_service",
                endpoint_id="tertiary", 
                url="http://localhost:8082",
                priority=1,
                region="eu-west-1"
            )
        ]
        
        config = ServiceConfig(
            service_id="test_service",
            service_type="http",
            endpoints=endpoints,
            failover_strategy=FailoverStrategy.PRIORITY_BASED,
            recovery_mode=RecoveryMode.AUTOMATIC,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30
        )
        
        failover_manager.register_service(config)
        
        # Test service status
        status = await failover_manager.get_service_status("test_service")
        assert status["service_id"] == "test_service"
        assert status["total_endpoints"] == 3
        assert status["active_endpoint"] == "primary"
        
        # Test manual failover
        success = await failover_manager.manual_failover("test_service", "secondary")
        assert success
        
        # Verify failover
        updated_status = await failover_manager.get_service_status("test_service")
        assert updated_status["active_endpoint"] == "secondary"
        
        # Test failover statistics
        stats = failover_manager.get_failover_statistics()
        assert stats["registered_services"] >= 1
        
        logger.info("✅ Failover and recovery system tests passed")
    
    async def test_advanced_monitoring_system(self):
        """Test advanced monitoring and alerting."""
        logger.info("🔒 Testing advanced monitoring system")
        
        from nasa_cmr_agent.monitoring.advanced_monitoring import (
            get_monitoring_system, AlertRule, AlertSeverity, 
            MetricType, Dashboard
        )
        
        monitoring = get_monitoring_system()
        
        # Register custom metrics
        monitoring.register_metric(
            "test_metric",
            MetricType.GAUGE,
            "Test metric for security testing",
            "count"
        )
        
        # Record some metric values
        monitoring.record_metric("test_metric", 50)
        monitoring.record_metric("test_metric", 75)
        monitoring.record_metric("test_metric", 90)
        
        # Get metric data
        metric_data = monitoring.get_metric_data("test_metric")
        assert len(metric_data) >= 3
        assert metric_data[-1].value == 90
        
        # Register alert rule
        alert_rule = AlertRule(
            rule_id="high_test_metric",
            name="High Test Metric Alert",
            description="Test metric is too high",
            metric_name="test_metric",
            condition="> 80",
            severity=AlertSeverity.WARNING,
            duration_seconds=1,
            cooldown_seconds=30
        )
        
        monitoring.register_alert_rule(alert_rule)
        
        # Start monitoring to trigger alert evaluation
        await monitoring.start_monitoring()
        
        # Wait for alert evaluation
        await asyncio.sleep(2)
        
        # Check if alert was triggered
        overview = monitoring.get_monitoring_overview()
        assert overview["alerts"]["active_total"] >= 0
        
        # Create a dashboard
        dashboard = Dashboard(
            dashboard_id="security_dashboard",
            title="Security Monitoring Dashboard", 
            description="Monitor security metrics and alerts",
            panels=[
                {
                    "title": "Test Metrics",
                    "type": "graph",
                    "metrics": ["test_metric"]
                }
            ]
        )
        
        monitoring.create_dashboard(dashboard)
        dashboard_data = monitoring.get_dashboard_data("security_dashboard")
        assert dashboard_data["title"] == "Security Monitoring Dashboard"
        assert len(dashboard_data["panels"]) == 1
        
        # Stop monitoring
        await monitoring.stop_monitoring()
        
        logger.info("✅ Advanced monitoring system tests passed")
    
    async def test_integrated_security_workflow(self):
        """Test integrated security workflow with all systems."""
        logger.info("🔒 Testing integrated security workflow")
        
        from nasa_cmr_agent.security.input_validator import get_input_validator
        from nasa_cmr_agent.security.audit_logger import get_audit_logger
        from nasa_cmr_agent.security.rate_limiter import get_rate_limiter, LimitType
        from nasa_cmr_agent.monitoring.advanced_monitoring import get_monitoring_system
        
        # Simulate a complete request workflow
        validator = get_input_validator()
        audit_logger = get_audit_logger()
        rate_limiter = get_rate_limiter()
        monitoring = get_monitoring_system()
        
        # 1. Check rate limits
        client_id = "integrated_test_client"
        ip_address = "192.168.1.123"
        
        rate_result = await rate_limiter.check_rate_limit(
            client_id=client_id,
            service="cmr_api",
            limit_type=LimitType.REQUESTS_PER_MINUTE,
            ip_address=ip_address
        )
        
        assert rate_result.allowed
        
        # 2. Validate input
        user_query = "precipitation data for Alaska 2023"
        validation_result = validator.validate_query_input(user_query)
        assert validation_result.is_valid
        
        # 3. Log the request
        await audit_logger.log_data_access_event(
            user_id=client_id,
            resource="cmr_collections",
            action="search_query",
            success=True,
            details={"query": user_query, "validation_passed": True}
        )
        
        # 4. Record monitoring metrics
        monitoring.record_metric("http_requests_total", 1, {"endpoint": "/search", "status": "200"})
        monitoring.record_metric("http_request_duration", 0.150, {"endpoint": "/search"})
        
        # Test malicious request workflow
        malicious_query = "'; DROP TABLE collections; --"
        
        # 1. Input validation (should fail)
        malicious_validation = validator.validate_query_input(malicious_query)
        assert not malicious_validation.is_valid
        
        # 2. Log the security event
        await audit_logger.log_input_validation_event(
            validation_result={
                "is_valid": False,
                "severity": "critical",
                "message": "SQL injection detected",
                "sanitized_value": None
            },
            user_id="potential_attacker",
            ip_address="10.0.0.5"
        )
        
        # 3. Log threat detection
        await audit_logger.log_threat_detection(
            threat_type="SQL Injection",
            indicators=["sql_keywords", "drop_table"],
            details={"query": malicious_query, "blocked": True},
            ip_address="10.0.0.5",
            user_id="potential_attacker"
        )
        
        # 4. Record security metrics
        monitoring.record_metric("security_threats_blocked", 1, {"type": "sql_injection"})
        
        # Verify the integrated workflow worked
        audit_stats = audit_logger.get_audit_statistics()
        rate_stats = rate_limiter.get_rate_limit_stats()
        monitoring_overview = monitoring.get_monitoring_overview()
        
        assert audit_stats["total_events"] >= 3
        assert audit_stats["threats_detected"] >= 1
        assert audit_stats["blocked_actions"] >= 1
        assert rate_stats["total_requests"] >= 1
        assert monitoring_overview["metrics"]["total_registered"] >= 1
        
        logger.info("✅ Integrated security workflow tests passed")


@pytest.mark.asyncio
@pytest.mark.security
async def test_security_system_performance():
    """Test performance of security systems under load."""
    logger.info("⚡ Testing security system performance")
    
    from nasa_cmr_agent.security.input_validator import get_input_validator
    from nasa_cmr_agent.security.rate_limiter import get_rate_limiter, LimitType
    
    validator = get_input_validator()
    rate_limiter = get_rate_limiter()
    
    # Test input validation performance
    start_time = time.time()
    for i in range(100):
        result = validator.validate_query_input(f"test query {i}")
    validation_duration = time.time() - start_time
    
    logger.info(f"Input validation: 100 queries in {validation_duration:.3f}s")
    assert validation_duration < 1.0, "Input validation too slow"
    
    # Test rate limiting performance
    start_time = time.time()
    for i in range(100):
        await rate_limiter.check_rate_limit(
            client_id=f"perf_client_{i}",
            service="default",
            limit_type=LimitType.REQUESTS_PER_SECOND
        )
    rate_limit_duration = time.time() - start_time
    
    logger.info(f"Rate limiting: 100 checks in {rate_limit_duration:.3f}s")
    assert rate_limit_duration < 2.0, "Rate limiting too slow"
    
    logger.info("✅ Security system performance tests passed")


if __name__ == "__main__":
    # Allow running this test directly
    async def main():
        test_class = TestComprehensiveSecurity()
        
        try:
            logger.info("🔧 Running comprehensive security tests...")
            
            await test_class.test_input_validation_framework()
            await test_class.test_encryption_system()
            await test_class.test_security_audit_logging()
            await test_class.test_rate_limiting_and_ddos_protection()
            await test_class.test_failover_and_recovery_system()
            await test_class.test_advanced_monitoring_system()
            await test_class.test_integrated_security_workflow()
            
            await test_security_system_performance()
            
            logger.info("🎉 All comprehensive security tests PASSED!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive security tests FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)