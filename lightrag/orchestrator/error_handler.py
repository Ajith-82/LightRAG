"""
Error handling and retry logic for Ollama batching orchestrator.

Provides robust error recovery mechanisms to maintain system reliability
during batch processing operations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)


class BatchProcessingErrorHandler:
    """
    Comprehensive error handler for batch processing operations.
    
    Features:
    - Exponential backoff retry logic
    - Circuit breaker pattern
    - Error categorization and handling
    - Performance monitoring during errors
    """
    
    def __init__(self, max_retries: int = 3, backoff_multiplier: float = 2.0):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_multiplier: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        
        # Error tracking
        self._error_stats = {
            "total_errors": 0,
            "retry_attempts": 0,
            "permanent_failures": 0,
            "recoveries": 0,
            "error_types": {}
        }
        
        # Circuit breaker state
        self._circuit_breaker = {
            "state": "closed",  # closed, open, half-open
            "failure_count": 0,
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
            "last_failure_time": 0
        }
        
        logger.info(f"Error handler initialized with max_retries={max_retries}")
    
    async def handle_batch_failure(
        self, 
        requests: List[Dict[str, Any]], 
        error: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Handle a failed batch of requests.
        
        Args:
            requests: List of requests that failed
            error: Error message or exception details
            context: Additional context about the failure
        """
        logger.error(f"Handling batch failure: {error}")
        
        # Update error statistics
        self._update_error_stats(error, len(requests))
        
        # Update circuit breaker state
        await self._update_circuit_breaker(error)
        
        # Categorize the error
        error_type = self._categorize_error(error)
        
        if error_type == "transient":
            # Handle transient errors with retry
            await self._handle_transient_error(requests, error, context)
        elif error_type == "timeout":
            # Handle timeout errors
            await self._handle_timeout_error(requests, error, context)
        elif error_type == "connection":
            # Handle connection errors
            await self._handle_connection_error(requests, error, context)
        elif error_type == "rate_limit":
            # Handle rate limiting
            await self._handle_rate_limit_error(requests, error, context)
        else:
            # Handle permanent errors
            await self._handle_permanent_error(requests, error, context)
    
    async def handle_request_retry(
        self, 
        request: Dict[str, Any], 
        error: str
    ) -> bool:
        """
        Handle retry logic for individual request.
        
        Args:
            request: Request that failed
            error: Error message
            
        Returns:
            True if request should be retried, False if permanently failed
        """
        retry_count = request.get("retry_count", 0)
        
        if retry_count >= self.max_retries:
            logger.error(f"Request {request['request_id']} exceeded max retries ({self.max_retries})")
            self._error_stats["permanent_failures"] += 1
            return False
        
        # Check circuit breaker
        if not await self._can_retry():
            logger.warning(f"Circuit breaker open, cannot retry request {request['request_id']}")
            return False
        
        # Calculate backoff delay
        delay = self._calculate_backoff_delay(retry_count)
        
        logger.info(f"Retrying request {request['request_id']} in {delay:.2f}s (attempt {retry_count + 1}/{self.max_retries})")
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        self._error_stats["retry_attempts"] += 1
        return True
    
    async def handle_timeout(self, batch_id: str, timeout_duration: float):
        """
        Handle batch timeout scenario.
        
        Args:
            batch_id: Identifier of the timed-out batch
            timeout_duration: How long the batch was processing before timeout
        """
        logger.warning(f"Batch {batch_id} timed out after {timeout_duration:.2f}s")
        
        # Update circuit breaker
        await self._update_circuit_breaker("timeout")
        
        # Implement timeout-specific recovery logic
        if timeout_duration > 30.0:
            # Very long timeout - likely system issue
            logger.error("Long timeout detected, triggering circuit breaker")
            self._circuit_breaker["state"] = "open"
            self._circuit_breaker["last_failure_time"] = time.time()
    
    async def handle_ollama_unavailable(self):
        """Handle Ollama service unavailability."""
        logger.error("Ollama service appears to be unavailable")
        
        # Open circuit breaker immediately
        self._circuit_breaker["state"] = "open"
        self._circuit_breaker["failure_count"] = self._circuit_breaker["failure_threshold"]
        self._circuit_breaker["last_failure_time"] = time.time()
        
        # Update error stats
        self._error_stats["total_errors"] += 1
        self._error_stats["error_types"]["service_unavailable"] = (
            self._error_stats["error_types"].get("service_unavailable", 0) + 1
        )
    
    async def _handle_transient_error(
        self, 
        requests: List[Dict[str, Any]], 
        error: str, 
        context: Optional[Dict[str, Any]]
    ):
        """Handle transient errors that can be retried."""
        logger.info(f"Handling transient error for {len(requests)} requests: {error}")
        
        # Individual retry for each request
        from .request_queue import RequestQueue
        
        # Note: This would need access to the request queue instance
        # For now, we'll log the action
        logger.info(f"Would requeue {len(requests)} requests for retry")
    
    async def _handle_timeout_error(
        self, 
        requests: List[Dict[str, Any]], 
        error: str, 
        context: Optional[Dict[str, Any]]
    ):
        """Handle timeout errors with exponential backoff."""
        logger.warning(f"Handling timeout error for {len(requests)} requests")
        
        # Add random jitter to prevent thundering herd
        base_delay = 1.0
        jitter = random.uniform(0.5, 1.5)
        delay = base_delay * jitter
        
        logger.info(f"Waiting {delay:.2f}s before requeuing timeout requests")
        await asyncio.sleep(delay)
    
    async def _handle_connection_error(
        self, 
        requests: List[Dict[str, Any]], 
        error: str, 
        context: Optional[Dict[str, Any]]
    ):
        """Handle connection errors."""
        logger.error(f"Handling connection error for {len(requests)} requests: {error}")
        
        # Longer delay for connection issues
        delay = 5.0 + random.uniform(0, 2.0)
        logger.info(f"Waiting {delay:.2f}s for connection recovery")
        await asyncio.sleep(delay)
    
    async def _handle_rate_limit_error(
        self, 
        requests: List[Dict[str, Any]], 
        error: str, 
        context: Optional[Dict[str, Any]]
    ):
        """Handle rate limiting errors."""
        logger.warning(f"Rate limit hit for {len(requests)} requests")
        
        # Significant delay for rate limiting
        delay = 10.0 + random.uniform(0, 5.0)
        logger.info(f"Waiting {delay:.2f}s for rate limit recovery")
        await asyncio.sleep(delay)
    
    async def _handle_permanent_error(
        self, 
        requests: List[Dict[str, Any]], 
        error: str, 
        context: Optional[Dict[str, Any]]
    ):
        """Handle permanent errors that cannot be retried."""
        logger.error(f"Permanent error for {len(requests)} requests: {error}")
        
        # Log failed requests for analysis
        for request in requests:
            logger.error(f"Permanently failed request {request['request_id']}: {request.get('text', '')[:100]}...")
        
        self._error_stats["permanent_failures"] += len(requests)
    
    def _categorize_error(self, error: str) -> str:
        """
        Categorize error type for appropriate handling.
        
        Args:
            error: Error message or exception details
            
        Returns:
            Error category: transient, timeout, connection, rate_limit, permanent
        """
        error_lower = error.lower()
        
        # Timeout errors
        if any(keyword in error_lower for keyword in ["timeout", "timed out", "deadline"]):
            return "timeout"
        
        # Connection errors
        if any(keyword in error_lower for keyword in ["connection", "network", "unreachable", "refused"]):
            return "connection"
        
        # Rate limiting
        if any(keyword in error_lower for keyword in ["rate limit", "too many", "quota", "throttle"]):
            return "rate_limit"
        
        # HTTP errors that might be transient
        if any(keyword in error_lower for keyword in ["502", "503", "504", "server error"]):
            return "transient"
        
        # Model or validation errors (permanent)
        if any(keyword in error_lower for keyword in ["invalid", "model not found", "bad request", "400"]):
            return "permanent"
        
        # Default to transient for unknown errors
        return "transient"
    
    def _calculate_backoff_delay(self, retry_count: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            retry_count: Current retry attempt (0-based)
            
        Returns:
            Delay in seconds
        """
        base_delay = 1.0
        exponential_delay = base_delay * (self.backoff_multiplier ** retry_count)
        
        # Add jitter to prevent thundering herd
        jitter_factor = random.uniform(0.5, 1.5)
        delay = exponential_delay * jitter_factor
        
        # Cap maximum delay
        max_delay = 60.0
        return min(delay, max_delay)
    
    async def _update_circuit_breaker(self, error: str):
        """Update circuit breaker state based on error."""
        current_time = time.time()
        
        if self._circuit_breaker["state"] == "open":
            # Check if we can transition to half-open
            if current_time - self._circuit_breaker["last_failure_time"] > self._circuit_breaker["recovery_timeout"]:
                self._circuit_breaker["state"] = "half-open"
                logger.info("Circuit breaker transitioning to half-open")
        
        elif self._circuit_breaker["state"] == "closed":
            # Increment failure count
            self._circuit_breaker["failure_count"] += 1
            
            # Check if we should open the circuit
            if self._circuit_breaker["failure_count"] >= self._circuit_breaker["failure_threshold"]:
                self._circuit_breaker["state"] = "open"
                self._circuit_breaker["last_failure_time"] = current_time
                logger.warning("Circuit breaker opened due to repeated failures")
        
        elif self._circuit_breaker["state"] == "half-open":
            # Failure in half-open state - back to open
            self._circuit_breaker["state"] = "open"
            self._circuit_breaker["failure_count"] += 1
            self._circuit_breaker["last_failure_time"] = current_time
            logger.warning("Circuit breaker back to open after half-open failure")
    
    async def _can_retry(self) -> bool:
        """Check if retries are allowed based on circuit breaker state."""
        if self._circuit_breaker["state"] == "open":
            # Check if recovery timeout has passed
            current_time = time.time()
            if current_time - self._circuit_breaker["last_failure_time"] > self._circuit_breaker["recovery_timeout"]:
                self._circuit_breaker["state"] = "half-open"
                logger.info("Circuit breaker allowing test request (half-open)")
                return True
            else:
                return False
        
        return True  # Closed or half-open allows retries
    
    def _update_error_stats(self, error: str, request_count: int):
        """Update error statistics."""
        self._error_stats["total_errors"] += 1
        
        error_type = self._categorize_error(error)
        self._error_stats["error_types"][error_type] = (
            self._error_stats["error_types"].get(error_type, 0) + 1
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        total_requests = max(
            self._error_stats["total_errors"] + self._error_stats["recoveries"],
            1
        )
        
        return {
            "total_errors": self._error_stats["total_errors"],
            "retry_attempts": self._error_stats["retry_attempts"],
            "permanent_failures": self._error_stats["permanent_failures"],
            "recoveries": self._error_stats["recoveries"],
            "error_rate": self._error_stats["total_errors"] / total_requests,
            "recovery_rate": self._error_stats["recoveries"] / max(self._error_stats["total_errors"], 1),
            "error_types": self._error_stats["error_types"],
            "circuit_breaker": {
                "state": self._circuit_breaker["state"],
                "failure_count": self._circuit_breaker["failure_count"],
                "last_failure_time": self._circuit_breaker["last_failure_time"]
            }
        }
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker to closed state."""
        self._circuit_breaker["state"] = "closed"
        self._circuit_breaker["failure_count"] = 0
        self._circuit_breaker["last_failure_time"] = 0
        logger.info("Circuit breaker manually reset to closed")
    
    def reset_stats(self):
        """Reset error statistics."""
        self._error_stats = {
            "total_errors": 0,
            "retry_attempts": 0,
            "permanent_failures": 0,
            "recoveries": 0,
            "error_types": {}
        }
        logger.info("Error handler statistics reset")