import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerService:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by stopping requests to failing services
    and allowing them time to recover.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0  # For half-open state
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def call(self):
        """Context manager for circuit breaker calls."""
        return CircuitBreakerContext(self)
    
    async def _can_execute(self) -> bool:
        """Check if request can be executed based on circuit state."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            elif self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (self.last_failure_time and 
                    datetime.now() - self.last_failure_time >= timedelta(seconds=self.recovery_timeout)):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True
            
            return False
    
    async def _on_success(self):
        """Handle successful execution."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                # If enough successes, close the circuit
                if self.success_count >= 3:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.last_failure_time = None
                    logger.info("Circuit breaker closed after successful recovery")
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        async with self._lock:
            if not isinstance(exception, self.expected_exception):
                return  # Don't count unexpected exceptions
            
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure during half-open means service still not recovered
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker opened again after failure in HALF_OPEN state")
            
            elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    async def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            self.success_count = 0
            logger.info("Circuit breaker manually reset to CLOSED state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recovery_timeout": self.recovery_timeout,
            "success_count": self.success_count if self.state == CircuitState.HALF_OPEN else None
        }


class CircuitBreakerContext:
    """Context manager for circuit breaker execution."""
    
    def __init__(self, circuit_breaker: CircuitBreakerService):
        self.circuit_breaker = circuit_breaker
    
    async def __aenter__(self):
        can_execute = await self.circuit_breaker._can_execute()
        if not can_execute:
            raise CircuitBreakerError("Circuit breaker is OPEN")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            await self.circuit_breaker._on_success()
        else:
            # Failure
            await self.circuit_breaker._on_failure(exc_val)
        return False  # Don't suppress exceptions