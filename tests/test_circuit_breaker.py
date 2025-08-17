"""
Unit tests for circuit breaker implementation.

Tests the circuit breaker pattern implementation for resilience.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock

from lightrag.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerError,
    CircuitBreakerTimeoutError,
    get_circuit_breaker,
    circuit_breaker,
    get_all_circuit_breakers,
    reset_all_circuit_breakers
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.expected_exceptions == (Exception,)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=15.0,
            expected_exceptions=(ValueError, ConnectionError)
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.timeout == 15.0
        assert config.expected_exceptions == (ValueError, ConnectionError)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)
        
        assert breaker.get_state() == CircuitBreakerState.CLOSED
        assert breaker.name == "test"
        assert breaker.config.failure_threshold == 2
    
    @pytest.mark.asyncio
    async def test_successful_calls(self):
        """Test successful function calls."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == CircuitBreakerState.CLOSED
        assert breaker.stats.total_successes == 1
        assert breaker.stats.total_calls == 1
    
    @pytest.mark.asyncio
    async def test_failure_counting(self):
        """Test failure counting and state transitions."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        breaker = CircuitBreaker("test", config)
        
        async def failing_func():
            raise ValueError("Test error")
        
        # First failure
        with pytest.raises(ValueError):
            await breaker.call(failing_func)
        
        assert breaker.get_state() == CircuitBreakerState.CLOSED
        assert breaker.stats.failure_count == 1
        
        # Second failure should open circuit
        with pytest.raises(ValueError):
            await breaker.call(failing_func)
        
        assert breaker.get_state() == CircuitBreakerState.OPEN
        assert breaker.stats.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_circuit_open_behavior(self):
        """Test circuit breaker behavior when open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker("test", config)
        
        async def failing_func():
            raise ValueError("Test error")
        
        # Trigger circuit to open
        with pytest.raises(ValueError):
            await breaker.call(failing_func)
        
        assert breaker.get_state() == CircuitBreakerState.OPEN
        
        # Subsequent calls should fail fast
        with pytest.raises(CircuitBreakerError):
            await breaker.call(failing_func)
        
        # Should not increment failure count
        assert breaker.stats.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_half_open_recovery(self):
        """Test recovery from OPEN to HALF_OPEN state."""
        config = CircuitBreakerConfig(
            failure_threshold=1, 
            recovery_timeout=0.05,  # Very short for testing
            success_threshold=2
        )
        breaker = CircuitBreaker("test", config)
        
        async def failing_func():
            raise ValueError("Test error")
        
        async def success_func():
            return "success"
        
        # Trigger circuit to open
        with pytest.raises(ValueError):
            await breaker.call(failing_func)
        
        assert breaker.get_state() == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.1)
        
        # Next call should put circuit in HALF_OPEN
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == CircuitBreakerState.HALF_OPEN
        
        # Another success should close the circuit
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.get_state() == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        config = CircuitBreakerConfig(timeout=0.1)
        breaker = CircuitBreaker("test", config)
        
        async def slow_func():
            await asyncio.sleep(0.2)  # Longer than timeout
            return "too slow"
        
        with pytest.raises(CircuitBreakerTimeoutError):
            await breaker.call(slow_func)
        
        assert breaker.stats.total_failures == 1
    
    def test_manual_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)
        
        # Simulate failures to open circuit
        breaker.stats.failure_count = 5
        breaker.state = CircuitBreakerState.OPEN
        
        # Reset should return to CLOSED
        breaker.reset()
        assert breaker.get_state() == CircuitBreakerState.CLOSED
        assert breaker.stats.failure_count == 0
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        stats = breaker.get_stats()
        
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["total_calls"] == 0
        assert "config" in stats
        assert stats["config"]["failure_threshold"] == 3


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator functionality."""
    
    @pytest.mark.asyncio
    async def test_decorator_usage(self):
        """Test using circuit breaker as decorator."""
        
        @circuit_breaker("test_decorator", failure_threshold=2)
        async def test_func(value):
            if value == "fail":
                raise ValueError("Test failure")
            return f"success: {value}"
        
        # Test successful call
        result = await test_func("good")
        assert result == "success: good"
        
        # Test failure
        with pytest.raises(ValueError):
            await test_func("fail")
        
        # Circuit should still be closed after one failure
        result = await test_func("good")
        assert result == "success: good"


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry functionality."""
    
    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        # Clear registry
        reset_all_circuit_breakers()
        
        # Create new breaker
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker1 = get_circuit_breaker("registry_test", config)
        
        assert breaker1.name == "registry_test"
        assert breaker1.config.failure_threshold == 3
        
        # Get existing breaker
        breaker2 = get_circuit_breaker("registry_test")
        
        assert breaker1 is breaker2  # Should be same instance
    
    def test_get_all_circuit_breakers(self):
        """Test getting all circuit breaker stats."""
        reset_all_circuit_breakers()
        
        # Create a few breakers
        get_circuit_breaker("test1")
        get_circuit_breaker("test2")
        
        all_breakers = get_all_circuit_breakers()
        
        assert len(all_breakers) >= 2
        assert "test1" in all_breakers
        assert "test2" in all_breakers
        
        for name, stats in all_breakers.items():
            assert "state" in stats
            assert "total_calls" in stats
            assert "config" in stats


class TestResilienceIntegration:
    """Test integration with resilience patterns."""
    
    @pytest.mark.asyncio
    async def test_llm_circuit_breaker(self):
        """Test LLM-specific circuit breaker."""
        from lightrag.circuit_breaker import get_llm_circuit_breaker
        
        breaker = get_llm_circuit_breaker("test_provider")
        
        assert breaker.name == "llm_test_provider"
        assert breaker.config.failure_threshold == 3  # LLM specific
        assert breaker.config.timeout == 60.0  # Longer timeout for LLM
    
    @pytest.mark.asyncio
    async def test_embedding_circuit_breaker(self):
        """Test embedding-specific circuit breaker."""
        from lightrag.circuit_breaker import get_embedding_circuit_breaker
        
        breaker = get_embedding_circuit_breaker("test_provider")
        
        assert breaker.name == "embedding_test_provider"
        assert breaker.config.failure_threshold == 5  # Higher threshold
        assert breaker.config.timeout == 30.0  # Standard timeout
    
    @pytest.mark.asyncio
    async def test_storage_circuit_breaker(self):
        """Test storage-specific circuit breaker."""
        from lightrag.circuit_breaker import get_storage_circuit_breaker
        
        breaker = get_storage_circuit_breaker("test_storage")
        
        assert breaker.name == "storage_test_storage"
        assert breaker.config.failure_threshold == 10  # Higher threshold
        assert breaker.config.timeout == 15.0  # Shorter timeout