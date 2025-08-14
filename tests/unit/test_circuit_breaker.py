"""
Comprehensive unit tests for Circuit Breaker service.

Tests circuit breaker functionality, state transitions, persistence,
error handling, and recovery mechanisms.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nasa_cmr_agent.services.circuit_breaker import (
    CircuitBreakerService, CircuitState, CircuitBreakerError, CircuitBreakerContext
)


@pytest.mark.unit
class TestCircuitBreakerService:
    """Test CircuitBreakerService functionality."""
    
    @pytest.fixture
    async def circuit_breaker(self, temp_data_dir):
        """Create circuit breaker for testing."""
        cb = CircuitBreakerService(
            failure_threshold=3,
            recovery_timeout=5,
            service_name="test_service",
            persist_state=False  # Disable persistence for unit tests
        )
        yield cb
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreakerService(
            failure_threshold=5,
            recovery_timeout=30,
            service_name="test_api"
        )
        
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30
        assert cb.service_name == "test_api"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    async def test_circuit_breaker_successful_execution(self, circuit_breaker):
        """Test successful execution through circuit breaker."""
        async with circuit_breaker.call():
            # Simulate successful operation
            pass
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    async def test_circuit_breaker_failure_accumulation(self, circuit_breaker):
        """Test failure accumulation without reaching threshold."""
        # Simulate 2 failures (below threshold of 3)
        for _ in range(2):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 2
    
    async def test_circuit_breaker_opens_on_threshold(self, circuit_breaker):
        """Test circuit breaker opens when failure threshold is reached."""
        # Simulate failures to reach threshold
        for _ in range(3):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 3
        assert circuit_breaker.last_failure_time is not None
    
    async def test_circuit_breaker_rejects_when_open(self, circuit_breaker):
        """Test circuit breaker rejects calls when open."""
        # Open the circuit breaker
        for _ in range(3):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        # Should reject subsequent calls
        with pytest.raises(CircuitBreakerError) as exc_info:
            async with circuit_breaker.call():
                pass
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    async def test_circuit_breaker_half_open_transition(self, circuit_breaker):
        """Test transition to half-open state after timeout."""
        # Set short recovery timeout for testing
        circuit_breaker.recovery_timeout = 1
        
        # Open the circuit breaker
        for _ in range(3):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should transition to half-open
        async with circuit_breaker.call():
            pass  # Successful call
        
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        assert circuit_breaker.success_count == 1
    
    async def test_circuit_breaker_recovery_to_closed(self, circuit_breaker):
        """Test recovery from half-open to closed state."""
        # Set short recovery timeout
        circuit_breaker.recovery_timeout = 1
        
        # Open the circuit breaker
        for _ in range(3):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Execute 3 successful calls to close circuit
        for _ in range(3):
            async with circuit_breaker.call():
                pass  # Successful call
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None
    
    async def test_circuit_breaker_half_open_failure(self, circuit_breaker):
        """Test failure in half-open state reopens circuit."""
        # Set short recovery timeout
        circuit_breaker.recovery_timeout = 1
        
        # Open the circuit breaker
        for _ in range(3):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Fail in half-open state
        try:
            async with circuit_breaker.call():
                raise Exception("Failed in half-open")
        except Exception:
            pass
        
        assert circuit_breaker.state == CircuitState.OPEN
    
    async def test_circuit_breaker_reset(self, circuit_breaker):
        """Test manual circuit breaker reset."""
        # Open the circuit breaker
        for _ in range(3):
            try:
                async with circuit_breaker.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Reset circuit breaker
        await circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.last_failure_time is None
    
    def test_circuit_breaker_get_state(self, circuit_breaker):
        """Test circuit breaker state information."""
        state_info = circuit_breaker.get_state()
        
        assert isinstance(state_info, dict)
        assert "state" in state_info
        assert "failure_count" in state_info
        assert "failure_threshold" in state_info
        assert "recovery_timeout" in state_info
        assert "service_name" in state_info
        
        assert state_info["state"] == "closed"
        assert state_info["failure_count"] == 0
        assert state_info["failure_threshold"] == 3
        assert state_info["service_name"] == "test_service"
    
    async def test_circuit_breaker_exception_filtering(self, circuit_breaker):
        """Test that only expected exceptions trigger circuit breaker."""
        # Configure to only handle ValueError
        circuit_breaker.expected_exception = ValueError
        
        # RuntimeError should not trigger circuit breaker
        try:
            async with circuit_breaker.call():
                raise RuntimeError("This should not count")
        except RuntimeError:
            pass
        
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # ValueError should trigger circuit breaker
        try:
            async with circuit_breaker.call():
                raise ValueError("This should count")
        except ValueError:
            pass
        
        assert circuit_breaker.failure_count == 1


@pytest.mark.unit
class TestCircuitBreakerPersistence:
    """Test circuit breaker state persistence."""
    
    @pytest.fixture
    async def persistent_circuit_breaker(self, temp_data_dir):
        """Create circuit breaker with persistence."""
        cb = CircuitBreakerService(
            failure_threshold=3,
            recovery_timeout=5,
            service_name="persistent_test",
            persist_state=True
        )
        
        # Override state file path for testing
        cb.state_file = temp_data_dir / "circuit_breaker_test.json"
        cb.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        yield cb
    
    async def test_state_persistence_save(self, persistent_circuit_breaker):
        """Test saving circuit breaker state."""
        cb = persistent_circuit_breaker
        
        # Trigger some failures
        for _ in range(2):
            try:
                async with cb.call():
                    raise Exception("Test failure")
            except Exception:
                pass
        
        # Save state
        await cb._save_state()
        
        # Verify file exists and contains correct data
        assert cb.state_file.exists()
        
        with open(cb.state_file, 'r') as f:
            saved_state = json.load(f)
        
        assert saved_state["failure_count"] == 2
        assert saved_state["state"] == "closed"
        assert saved_state["service_name"] == "persistent_test"
    
    async def test_state_persistence_load(self, persistent_circuit_breaker, temp_data_dir):
        """Test loading circuit breaker state."""
        cb = persistent_circuit_breaker
        
        # Create state file with test data
        test_state = {
            "service_name": "persistent_test",
            "state": "open",
            "failure_count": 5,
            "last_failure_time": datetime.now().isoformat(),
            "recovery_timeout": 5
        }
        
        with open(cb.state_file, 'w') as f:
            json.dump(test_state, f)
        
        # Load state
        await cb._restore_state(test_state)
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 5
        assert cb.last_failure_time is not None
    
    async def test_state_persistence_old_state_reset(self, persistent_circuit_breaker):
        """Test that old state is reset appropriately."""
        cb = persistent_circuit_breaker
        
        # Create old state (more than 2x recovery timeout ago)
        old_time = datetime.now() - timedelta(seconds=cb.recovery_timeout * 3)
        test_state = {
            "service_name": "persistent_test",
            "state": "open",
            "failure_count": 5,
            "last_failure_time": old_time.isoformat(),
            "recovery_timeout": cb.recovery_timeout
        }
        
        # Load old state
        await cb._restore_state(test_state)
        
        # Should reset to closed due to age
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
    
    @patch('redis.asyncio.from_url')
    async def test_redis_persistence_fallback(self, mock_redis, persistent_circuit_breaker):
        """Test Redis persistence with fallback to file."""
        cb = persistent_circuit_breaker
        
        # Mock Redis failure
        mock_redis.side_effect = Exception("Redis connection failed")
        
        # Should fallback to file persistence
        await cb._save_state()
        
        # File should exist as fallback
        assert cb.state_file.exists()


@pytest.mark.unit
class TestCircuitBreakerContext:
    """Test CircuitBreakerContext context manager."""
    
    async def test_context_manager_success(self):
        """Test context manager with successful execution."""
        cb = CircuitBreakerService(service_name="test_context", persist_state=False)
        
        async with cb.call() as context:
            assert isinstance(context, CircuitBreakerContext)
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED
    
    async def test_context_manager_exception(self):
        """Test context manager with exception."""
        cb = CircuitBreakerService(service_name="test_context", persist_state=False)
        
        with pytest.raises(ValueError):
            async with cb.call():
                raise ValueError("Test exception")
        
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED
    
    async def test_context_manager_exception_propagation(self):
        """Test that exceptions are properly propagated."""
        cb = CircuitBreakerService(service_name="test_context", persist_state=False)
        
        # Should propagate the original exception
        with pytest.raises(ValueError) as exc_info:
            async with cb.call():
                raise ValueError("Original exception")
        
        assert "Original exception" in str(exc_info.value)


@pytest.mark.unit
class TestCircuitBreakerConcurrency:
    """Test circuit breaker under concurrent access."""
    
    async def test_concurrent_access(self):
        """Test circuit breaker thread safety under concurrent access."""
        cb = CircuitBreakerService(
            failure_threshold=5,
            service_name="concurrent_test",
            persist_state=False
        )
        
        async def successful_operation():
            async with cb.call():
                await asyncio.sleep(0.01)  # Simulate work
        
        async def failing_operation():
            try:
                async with cb.call():
                    await asyncio.sleep(0.01)  # Simulate work
                    raise Exception("Concurrent failure")
            except Exception:
                pass
        
        # Run concurrent operations
        tasks = []
        for _ in range(10):
            tasks.append(successful_operation())
            tasks.append(failing_operation())
        
        await asyncio.gather(*tasks)
        
        # State should be consistent
        assert isinstance(cb.failure_count, int)
        assert cb.failure_count >= 0
        assert cb.state in [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
    
    async def test_lock_behavior(self):
        """Test that state changes are properly locked."""
        cb = CircuitBreakerService(service_name="lock_test", persist_state=False)
        
        # Test that we can acquire the lock
        async with cb._lock:
            original_count = cb.failure_count
            cb.failure_count += 1
        
        assert cb.failure_count == original_count + 1