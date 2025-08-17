"""
Resilience utilities for LightRAG.

Provides easy integration of circuit breakers and other resilience patterns
with existing LightRAG components.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from .circuit_breaker import (
    CircuitBreakerConfig,
    get_circuit_breaker,
    get_llm_circuit_breaker,
    get_embedding_circuit_breaker,
    get_storage_circuit_breaker,
    CircuitBreakerError,
    CircuitBreakerTimeoutError
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResilienceManager:
    """Manages resilience patterns for LightRAG components."""
    
    def __init__(self):
        self._enabled = True
        self._fallback_handlers: Dict[str, Callable] = {}
    
    def enable(self):
        """Enable resilience patterns."""
        self._enabled = True
        logger.info("Resilience patterns enabled")
    
    def disable(self):
        """Disable resilience patterns."""
        self._enabled = False
        logger.warning("Resilience patterns disabled")
    
    def is_enabled(self) -> bool:
        """Check if resilience patterns are enabled."""
        return self._enabled
    
    def register_fallback(self, service_name: str, handler: Callable):
        """Register fallback handler for a service."""
        self._fallback_handlers[service_name] = handler
        logger.info(f"Registered fallback handler for {service_name}")
    
    def get_fallback(self, service_name: str) -> Optional[Callable]:
        """Get fallback handler for a service."""
        return self._fallback_handlers.get(service_name)


# Global resilience manager
_resilience_manager = ResilienceManager()


def with_llm_resilience(
    provider: str = "default",
    enable_fallback: bool = True,
    fallback_response: str = "I apologize, but I'm experiencing technical difficulties. Please try again later."
):
    """Add resilience to LLM functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not _resilience_manager.is_enabled():
                return await func(*args, **kwargs)
            
            circuit_breaker = get_llm_circuit_breaker(provider)
            
            try:
                return await circuit_breaker.call(func, *args, **kwargs)
                
            except (CircuitBreakerError, CircuitBreakerTimeoutError) as e:
                logger.error(f"LLM resilience failure for {provider}: {e}")
                
                if enable_fallback:
                    # Try fallback handler first
                    fallback = _resilience_manager.get_fallback(f"llm_{provider}")
                    if fallback:
                        try:
                            return await fallback(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback handler failed: {fallback_error}")
                    
                    # Use default fallback response
                    if isinstance(fallback_response, str):
                        return fallback_response
                
                raise
        
        return wrapper
    return decorator


def with_embedding_resilience(
    provider: str = "default",
    enable_fallback: bool = True,
    fallback_embeddings: Optional[Any] = None
):
    """Add resilience to embedding functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not _resilience_manager.is_enabled():
                return await func(*args, **kwargs)
            
            circuit_breaker = get_embedding_circuit_breaker(provider)
            
            try:
                return await circuit_breaker.call(func, *args, **kwargs)
                
            except (CircuitBreakerError, CircuitBreakerTimeoutError) as e:
                logger.error(f"Embedding resilience failure for {provider}: {e}")
                
                if enable_fallback:
                    # Try fallback handler first
                    fallback = _resilience_manager.get_fallback(f"embedding_{provider}")
                    if fallback:
                        try:
                            return await fallback(*args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback handler failed: {fallback_error}")
                    
                    # Use default fallback embeddings
                    if fallback_embeddings is not None:
                        return fallback_embeddings
                
                raise
        
        return wrapper
    return decorator


def with_storage_resilience(
    storage_type: str = "default",
    enable_retry: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """Add resilience to storage functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not _resilience_manager.is_enabled():
                return await func(*args, **kwargs)
            
            circuit_breaker = get_storage_circuit_breaker(storage_type)
            
            for attempt in range(max_retries + 1):
                try:
                    return await circuit_breaker.call(func, *args, **kwargs)
                    
                except (CircuitBreakerError, CircuitBreakerTimeoutError) as e:
                    if attempt == max_retries:
                        logger.error(f"Storage resilience failure for {storage_type} after {max_retries} retries: {e}")
                        raise
                    
                    if enable_retry:
                        logger.warning(f"Storage operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {retry_delay}s")
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0,
    expected_exceptions: tuple = (Exception,)
):
    """Generic circuit breaker decorator."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exceptions=expected_exceptions
        )
        circuit_breaker = get_circuit_breaker(name, config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not _resilience_manager.is_enabled():
                return await func(*args, **kwargs)
            
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


class ResilienceMixin:
    """Mixin class to add resilience to existing classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resilience_enabled = True
    
    def enable_resilience(self):
        """Enable resilience for this instance."""
        self._resilience_enabled = True
    
    def disable_resilience(self):
        """Disable resilience for this instance."""
        self._resilience_enabled = False
    
    def with_resilience(self, service_type: str, service_name: str = "default"):
        """Decorator to add resilience to methods."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                if not self._resilience_enabled or not _resilience_manager.is_enabled():
                    return await func(*args, **kwargs)
                
                if service_type == "llm":
                    decorator_func = with_llm_resilience(service_name)
                elif service_type == "embedding":
                    decorator_func = with_embedding_resilience(service_name)
                elif service_type == "storage":
                    decorator_func = with_storage_resilience(service_name)
                else:
                    decorator_func = with_circuit_breaker(f"{service_type}_{service_name}")
                
                wrapped_func = decorator_func(func)
                return await wrapped_func(*args, **kwargs)
            
            return wrapper
        return decorator


def get_resilience_status() -> Dict[str, Any]:
    """Get overall resilience status."""
    from .circuit_breaker import get_all_circuit_breakers
    
    circuit_breakers = get_all_circuit_breakers()
    
    unhealthy_count = sum(
        1 for stats in circuit_breakers.values() 
        if stats["state"] != "closed"
    )
    
    return {
        "enabled": _resilience_manager.is_enabled(),
        "circuit_breakers": circuit_breakers,
        "total_breakers": len(circuit_breakers),
        "unhealthy_breakers": unhealthy_count,
        "health_percentage": (
            ((len(circuit_breakers) - unhealthy_count) / len(circuit_breakers) * 100)
            if circuit_breakers else 100
        )
    }


def enable_resilience():
    """Enable global resilience patterns."""
    _resilience_manager.enable()


def disable_resilience():
    """Disable global resilience patterns."""
    _resilience_manager.disable()


def register_fallback_handler(service_name: str, handler: Callable):
    """Register a fallback handler for a service."""
    _resilience_manager.register_fallback(service_name, handler)


# Example fallback handlers
async def llm_cache_fallback(*args, **kwargs) -> str:
    """Fallback to cached responses when LLM is unavailable."""
    # This would integrate with the LLM cache to find similar queries
    logger.info("Using LLM cache fallback")
    return "I'm experiencing connectivity issues. This response is from cache."


async def embedding_zero_fallback(texts, **kwargs):
    """Fallback to zero embeddings when embedding service is unavailable."""
    import numpy as np
    logger.warning("Using zero embedding fallback")
    embedding_dim = kwargs.get('embedding_dim', 1536)  # Default OpenAI dimension
    return np.zeros((len(texts) if isinstance(texts, list) else 1, embedding_dim))


# Pre-register common fallback handlers
def setup_default_fallbacks():
    """Setup default fallback handlers."""
    register_fallback_handler("llm_openai", llm_cache_fallback)
    register_fallback_handler("llm_anthropic", llm_cache_fallback)
    register_fallback_handler("llm_ollama", llm_cache_fallback)
    register_fallback_handler("embedding_openai", embedding_zero_fallback)
    register_fallback_handler("embedding_ollama", embedding_zero_fallback)