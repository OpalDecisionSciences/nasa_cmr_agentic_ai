import asyncio
import json
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional
from enum import Enum
from pathlib import Path
import structlog
from redis import asyncio as aioredis

from ..core.config import settings

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
    Circuit breaker implementation with persistent state for fault tolerance.
    
    Prevents cascading failures by stopping requests to failing services
    and allowing them time to recover. State persists across service restarts.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: type = Exception,
        service_name: str = "default",
        persist_state: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.service_name = service_name
        self.persist_state = persist_state
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0  # For half-open state
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Persistence
        self.state_file = Path(f"data/circuit_breakers/{service_name}_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.redis_key = f"circuit_breaker:{service_name}"
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Load persisted state on initialization
        asyncio.create_task(self._load_state())
    
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
            state_changed = False
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                # If enough successes, close the circuit
                if self.success_count >= 3:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.last_failure_time = None
                    logger.info("Circuit breaker closed after successful recovery")
                    state_changed = True
            
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                if self.failure_count > 0:
                    self.failure_count = 0
                    state_changed = True
        
        # Save state if changed
        if state_changed:
            await self._save_state()
    
    async def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        async with self._lock:
            if not isinstance(exception, self.expected_exception):
                return  # Don't count unexpected exceptions
            
            state_changed = False
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure during half-open means service still not recovered
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker opened again after failure in HALF_OPEN state")
                state_changed = True
            
            elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                state_changed = True
            else:
                # Still accumulating failures
                state_changed = True
        
        # Save state after failure
        if state_changed:
            await self._save_state()
    
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
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recovery_timeout": self.recovery_timeout,
            "success_count": self.success_count if self.state == CircuitState.HALF_OPEN else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _load_state(self):
        """Load persisted state from storage."""
        if not self.persist_state:
            return
        
        state_loaded = False
        
        # Try loading from Redis first
        try:
            if not self.redis_client:
                self.redis_client = await aioredis.from_url(
                    settings.redis_url,
                    db=settings.redis_db,
                    decode_responses=True
                )
            
            state_json = await self.redis_client.get(self.redis_key)
            if state_json:
                state_data = json.loads(state_json)
                await self._restore_state(state_data)
                state_loaded = True
                logger.info(f"Loaded circuit breaker state from Redis: {self.service_name}")
        except Exception as e:
            logger.warning(f"Failed to load state from Redis: {e}")
        
        # Fall back to file if Redis failed
        if not state_loaded and self.state_file.exists():
            try:
                async with aiofiles.open(self.state_file, 'r') as f:
                    content = await f.read()
                    state_data = json.loads(content)
                    await self._restore_state(state_data)
                    logger.info(f"Loaded circuit breaker state from file: {self.service_name}")
            except Exception as e:
                logger.error(f"Failed to load state from file: {e}")
    
    async def _save_state(self):
        """Save current state to persistent storage."""
        if not self.persist_state:
            return
        
        state_data = self.get_state()
        state_json = json.dumps(state_data, default=str)
        
        # Save to Redis
        try:
            if self.redis_client:
                await self.redis_client.set(
                    self.redis_key,
                    state_json,
                    ex=86400  # 24 hour TTL
                )
        except Exception as e:
            logger.warning(f"Failed to save state to Redis: {e}")
        
        # Always save to file as backup
        try:
            async with aiofiles.open(self.state_file, 'w') as f:
                await f.write(state_json)
        except Exception as e:
            logger.error(f"Failed to save state to file: {e}")
    
    async def _restore_state(self, state_data: Dict[str, Any]):
        """Restore state from persisted data."""
        async with self._lock:
            # Check if state is still relevant (not too old)
            if state_data.get("last_failure_time"):
                last_failure = datetime.fromisoformat(state_data["last_failure_time"])
                time_since_failure = datetime.now() - last_failure
                
                # If it's been more than 2x recovery timeout, reset to closed
                if time_since_failure.total_seconds() > self.recovery_timeout * 2:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.last_failure_time = None
                    logger.info(f"Circuit breaker state too old, resetting: {self.service_name}")
                    return
            
            # Restore state
            self.state = CircuitState(state_data.get("state", "closed"))
            self.failure_count = state_data.get("failure_count", 0)
            self.success_count = state_data.get("success_count", 0)
            
            if state_data.get("last_failure_time"):
                self.last_failure_time = datetime.fromisoformat(state_data["last_failure_time"])


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