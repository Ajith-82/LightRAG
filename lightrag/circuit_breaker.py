"""
Circuit breaker pattern implementation for LightRAG.

Provides resilience mechanisms to prevent cascading failures when external
services (LLM APIs, databases, vector stores) become unavailable or unreliable.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit open, calls fail fast
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5           # Failures to trigger OPEN state
    recovery_timeout: float = 60.0       # Seconds before trying HALF_OPEN
    success_threshold: int = 3           # Successes to close from HALF_OPEN
    timeout: float = 30.0                # Request timeout
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    state_changed_time: float = field(default_factory=time.time)
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerTimeoutError(Exception):
    """Raised when operation times out."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for resilience patterns.
    
    Prevents cascading failures by monitoring service health and 
    failing fast when services are down.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator to wrap async functions with circuit breaker."""
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)
        return wrapper
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.stats.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker {self.name}: Attempting recovery (HALF_OPEN)")
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.stats.state_changed_time = time.time()
                else:
                    logger.warning(f"Circuit breaker {self.name}: OPEN - failing fast")
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
        
        # Execute the function with timeout
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.timeout
            )
            await self._on_success()
            return result
            
        except asyncio.TimeoutError:
            error = CircuitBreakerTimeoutError(
                f"Operation timed out after {self.config.timeout}s"
            )
            await self._on_failure(error)
            raise error
            
        except self.config.expected_exceptions as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self.stats.success_count += 1
            self.stats.total_successes += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit breaker {self.name}: Recovery successful (CLOSED)")
                    self.state = CircuitBreakerState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
                    self.stats.state_changed_time = time.time()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.stats.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        with self._lock:
            self.stats.failure_count += 1
            self.stats.total_failures += 1
            self.stats.last_failure_time = time.time()
            
            logger.warning(
                f"Circuit breaker {self.name}: Failure #{self.stats.failure_count} - {exception}"
            )
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.stats.failure_count >= self.config.failure_threshold:
                    logger.error(f"Circuit breaker {self.name}: Opening circuit after {self.stats.failure_count} failures")
                    self.state = CircuitBreakerState.OPEN
                    self.stats.state_changed_time = time.time()
                    
            elif self.state == CircuitBreakerState.HALF_OPEN:
                logger.error(f"Circuit breaker {self.name}: Recovery failed, returning to OPEN")
                self.state = CircuitBreakerState.OPEN
                self.stats.success_count = 0
                self.stats.state_changed_time = time.time()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        return (time.time() - self.stats.state_changed_time) >= self.config.recovery_timeout
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.stats.failure_count,
                "success_count": self.stats.success_count,
                "total_calls": self.stats.total_calls,
                "total_failures": self.stats.total_failures,
                "total_successes": self.stats.total_successes,
                "last_failure_time": self.stats.last_failure_time,
                "state_changed_time": self.stats.state_changed_time,
                "uptime": time.time() - self.stats.state_changed_time,
                "failure_rate": (
                    self.stats.total_failures / self.stats.total_calls 
                    if self.stats.total_calls > 0 else 0
                ),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                }
            }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Circuit breaker {self.name}: Manual reset to CLOSED")
            self.state = CircuitBreakerState.CLOSED
            self.stats.failure_count = 0
            self.stats.success_count = 0
            self.stats.state_changed_time = time.time()


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                if config is None:
                    config = CircuitBreakerConfig()
                self._breakers[name] = CircuitBreaker(name, config)
                logger.info(f"Created circuit breaker: {name}")
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def list_breakers(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")
    
    def get_unhealthy_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get circuit breakers that are not in CLOSED state."""
        with self._lock:
            return {
                name: breaker 
                for name, breaker in self._breakers.items()
                if breaker.get_state() != CircuitBreakerState.CLOSED
            }


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    return _registry.get_or_create(name, config)


def circuit_breaker(
    name: str, 
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0,
    expected_exceptions: tuple = (Exception,)
):
    """Decorator for circuit breaker protection."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exceptions=expected_exceptions
    )
    breaker = get_circuit_breaker(name, config)
    return breaker


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get stats for all circuit breakers."""
    return _registry.list_breakers()


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    _registry.reset_all()


# Pre-configured circuit breakers for common LightRAG services
def get_llm_circuit_breaker(provider: str = "default") -> CircuitBreaker:
    """Get circuit breaker for LLM provider."""
    config = CircuitBreakerConfig(
        failure_threshold=3,        # Lower threshold for LLM APIs
        recovery_timeout=120.0,     # Longer recovery time
        success_threshold=2,        # Fewer successes needed
        timeout=60.0,               # Longer timeout for LLM calls
        expected_exceptions=(Exception,)
    )
    return get_circuit_breaker(f"llm_{provider}", config)


def get_embedding_circuit_breaker(provider: str = "default") -> CircuitBreaker:
    """Get circuit breaker for embedding provider."""
    config = CircuitBreakerConfig(
        failure_threshold=5,        # Higher threshold for embeddings
        recovery_timeout=60.0,      # Standard recovery time
        success_threshold=3,        # Standard success threshold
        timeout=30.0,               # Standard timeout
        expected_exceptions=(Exception,)
    )
    return get_circuit_breaker(f"embedding_{provider}", config)


def get_storage_circuit_breaker(storage_type: str = "default") -> CircuitBreaker:
    """Get circuit breaker for storage operations."""
    config = CircuitBreakerConfig(
        failure_threshold=10,       # Higher threshold for storage
        recovery_timeout=30.0,      # Faster recovery for storage
        success_threshold=5,        # More successes needed
        timeout=15.0,               # Shorter timeout for storage
        expected_exceptions=(Exception,)
    )
    return get_circuit_breaker(f"storage_{storage_type}", config)