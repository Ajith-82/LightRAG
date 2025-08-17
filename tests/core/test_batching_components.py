"""
Unit tests for individual batching orchestrator components.

Tests each component in isolation to ensure correct functionality
of the request queue, cache, batch processor, and error handler.
"""

import asyncio
import json
import pickle
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# Import batching components
try:
    from lightrag.orchestrator.request_queue import RequestQueue
    from lightrag.orchestrator.embedding_cache import EmbeddingCache
    from lightrag.orchestrator.batch_processor import BatchProcessor
    from lightrag.orchestrator.error_handler import BatchProcessingErrorHandler
    from lightrag.config.batch_config import BatchingConfig
except ImportError:
    pytest.skip("Batching components not available", allow_module_level=True)


@pytest.mark.unit
@pytest.mark.batching
class TestRequestQueue:
    """Unit tests for request queue component."""
    
    @pytest.mark.asyncio
    async def test_initialization_with_redis(self, mock_redis_batching):
        """Test queue initialization with Redis."""
        queue = RequestQueue(mock_redis_batching)
        await queue.initialize()
        
        assert queue._initialized is True
        assert queue.redis is not None
        mock_redis_batching.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_without_redis(self):
        """Test queue initialization without Redis (in-memory fallback)."""
        queue = RequestQueue(None)
        await queue.initialize()
        
        assert queue._initialized is True
        assert queue.redis is None
    
    @pytest.mark.asyncio
    async def test_enqueue_request_redis(self, mock_redis_batching):
        """Test request enqueueing with Redis."""
        queue = RequestQueue(mock_redis_batching)
        await queue.initialize()
        
        request_data = {
            "request_id": "test-123",
            "text": "test text",
            "text_hash": "hash123",
            "timestamp": time.time(),
            "priority": 2,
            "retry_count": 0
        }
        
        await queue.enqueue_request(request_data)
        
        # Verify Redis lpush was called with serialized data
        mock_redis_batching.lpush.assert_called_once()
        call_args = mock_redis_batching.lpush.call_args
        assert call_args[0][0] == queue.QUEUE_KEYS['normal_priority']  # Queue key
        
        # Verify the serialized data can be deserialized
        serialized_data = call_args[0][1]
        deserialized = json.loads(serialized_data)
        assert deserialized["request_id"] == "test-123"
    
    @pytest.mark.asyncio
    async def test_enqueue_request_memory(self):
        """Test request enqueueing with in-memory fallback."""
        queue = RequestQueue(None)
        await queue.initialize()
        
        request_data = {
            "request_id": "test-456",
            "text": "test text",
            "priority": 1  # High priority
        }
        
        await queue.enqueue_request(request_data)
        
        # Verify it was added to in-memory queue
        assert len(queue._in_memory_queues['high_priority']) == 1
        assert queue._in_memory_queues['high_priority'][0]["request_id"] == "test-456"
    
    @pytest.mark.asyncio
    async def test_dequeue_batch_redis(self, mock_redis_batching):
        """Test batch dequeuing with Redis."""
        queue = RequestQueue(mock_redis_batching)
        await queue.initialize()
        
        # Mock Redis pipeline response
        mock_requests = [
            '{"request_id": "req1", "text": "text1", "priority": 2}',
            '{"request_id": "req2", "text": "text2", "priority": 2}'
        ]
        # Mock pipeline execution to return requests for each priority queue
        mock_redis_batching.execute.return_value = [mock_requests, [], []]  # high, normal, low priority queues
        
        batch = await queue.dequeue_batch(5)
        
        assert len(batch) == 2
        assert batch[0]["request_id"] == "req1"
        assert batch[1]["request_id"] == "req2"
    
    @pytest.mark.asyncio
    async def test_dequeue_batch_memory(self):
        """Test batch dequeuing with in-memory fallback."""
        queue = RequestQueue(None)
        await queue.initialize()
        
        # Add test requests to different priority queues
        high_priority_req = {"request_id": "high1", "priority": 1}
        normal_priority_req = {"request_id": "normal1", "priority": 2}
        low_priority_req = {"request_id": "low1", "priority": 3}
        
        await queue.enqueue_request(high_priority_req)
        await queue.enqueue_request(normal_priority_req)
        await queue.enqueue_request(low_priority_req)
        
        # Dequeue batch - should prioritize high priority first
        batch = await queue.dequeue_batch(2)
        
        assert len(batch) == 2
        assert batch[0]["request_id"] == "high1"  # High priority first
        assert batch[1]["request_id"] == "normal1"  # Normal priority second
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that requests are processed in priority order."""
        queue = RequestQueue(None)
        await queue.initialize()
        
        # Add requests in non-priority order
        requests = [
            {"request_id": "low1", "priority": 3},
            {"request_id": "high1", "priority": 1},
            {"request_id": "normal1", "priority": 2},
            {"request_id": "high2", "priority": 1}
        ]
        
        for req in requests:
            await queue.enqueue_request(req)
        
        # Dequeue all
        batch = await queue.dequeue_batch(10)
        
        # Should be ordered: high1, high2, normal1, low1
        assert len(batch) == 4
        assert batch[0]["request_id"] == "high1"
        assert batch[1]["request_id"] == "high2"
        assert batch[2]["request_id"] == "normal1"
        assert batch[3]["request_id"] == "low1"
    
    @pytest.mark.asyncio
    async def test_requeue_failed_requests(self, mock_redis_batching):
        """Test requeuing of failed requests."""
        queue = RequestQueue(mock_redis_batching)
        await queue.initialize()
        
        failed_requests = [
            {"request_id": "fail1", "text": "text1", "retry_count": 0, "priority": 2},
            {"request_id": "fail2", "text": "text2", "retry_count": 1, "priority": 1}
        ]
        
        await queue.requeue_failed_requests(failed_requests)
        
        # Should call lpush twice (once for each request)
        assert mock_redis_batching.lpush.call_count == 2
        
        # Verify retry count was incremented and priority adjusted
        for call_args in mock_redis_batching.lpush.call_args_list:
            serialized_data = call_args[0][1]
            request = json.loads(serialized_data)
            
            if request["request_id"] == "fail1":
                assert request["retry_count"] == 1
                assert request["priority"] == 3  # Demoted priority
            elif request["request_id"] == "fail2":
                assert request["retry_count"] == 2
                assert request["priority"] == 2  # Demoted priority
    
    @pytest.mark.asyncio
    async def test_queue_size_tracking(self, mock_redis_batching):
        """Test queue size tracking functionality."""
        queue = RequestQueue(mock_redis_batching)
        await queue.initialize()
        
        # Mock Redis llen responses
        mock_redis_batching.llen.side_effect = [5, 3, 2, 1]  # Different queue sizes
        
        sizes = await queue.get_queue_sizes()
        
        assert "high_priority" in sizes
        assert "normal_priority" in sizes
        assert "low_priority" in sizes
        assert "processing" in sizes
        
        total_size = await queue.size()
        assert total_size == 10  # Sum of all queue lengths (5+3+2+0 from mock responses)


@pytest.mark.unit
@pytest.mark.batching
class TestEmbeddingCache:
    """Unit tests for embedding cache component."""
    
    @pytest.mark.asyncio
    async def test_initialization_with_redis(self, mock_redis_batching):
        """Test cache initialization with Redis."""
        cache = EmbeddingCache(mock_redis_batching, db=2, ttl=3600)
        await cache.initialize()
        
        assert cache._initialized is True
        assert cache.redis is not None
        assert cache.db == 2
        assert cache.ttl == 3600
        mock_redis_batching.select.assert_called_once_with(2)
    
    @pytest.mark.asyncio
    async def test_initialization_without_redis(self):
        """Test cache initialization without Redis."""
        cache = EmbeddingCache(None)
        await cache.initialize()
        
        assert cache._initialized is True
        assert cache.redis is None
    
    @pytest.mark.asyncio
    async def test_text_hashing(self, mock_redis_batching):
        """Test consistent text hashing."""
        cache = EmbeddingCache(mock_redis_batching)
        await cache.initialize()
        
        text = "Test text for hashing"
        hash1 = await cache.get_text_hash(text)
        hash2 = await cache.get_text_hash(text)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        assert isinstance(hash1, str)
        
        # Different text should produce different hash
        different_text = "Different test text"
        hash3 = await cache.get_text_hash(different_text)
        assert hash1 != hash3
    
    @pytest.mark.asyncio
    async def test_cache_miss_redis(self, mock_redis_batching):
        """Test cache miss with Redis."""
        cache = EmbeddingCache(mock_redis_batching)
        await cache.initialize()
        
        # Mock Redis to return None (cache miss)
        mock_redis_batching.get.return_value = None
        
        result = await cache.get_cached_embedding("nonexistent_hash")
        
        assert result is None
        mock_redis_batching.get.assert_called_once()
        assert cache._cache_stats["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_hit_redis(self, mock_redis_batching):
        """Test cache hit with Redis."""
        cache = EmbeddingCache(mock_redis_batching)
        await cache.initialize()
        
        # Create test embedding
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        serialized_embedding = pickle.dumps(test_embedding)
        
        # Mock Redis to return serialized embedding
        mock_redis_batching.get.return_value = serialized_embedding
        
        result = await cache.get_cached_embedding("test_hash")
        
        assert result is not None
        np.testing.assert_array_equal(result, test_embedding)
        assert cache._cache_stats["hits"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_store_redis(self, mock_redis_batching):
        """Test storing embedding in Redis cache."""
        cache = EmbeddingCache(mock_redis_batching, ttl=1800)
        await cache.initialize()
        
        text_hash = "test_hash_123"
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        await cache.cache_embedding(text_hash, embedding)
        
        # Verify Redis setex was called with correct parameters
        mock_redis_batching.setex.assert_called_once()
        call_args = mock_redis_batching.setex.call_args
        
        assert call_args[0][0] == f"embedding:{text_hash}"  # Key
        assert call_args[0][1] == 1800  # TTL
        # The value should be pickled embedding
        unpickled = pickle.loads(call_args[0][2])
        np.testing.assert_array_equal(unpickled, embedding)
        
        assert cache._cache_stats["stores"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_memory_fallback(self):
        """Test in-memory cache fallback."""
        cache = EmbeddingCache(None, ttl=60)
        await cache.initialize()
        
        text_hash = "memory_test_hash"
        embedding = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        
        # Store in memory cache
        await cache.cache_embedding(text_hash, embedding)
        
        # Retrieve from memory cache
        result = await cache.get_cached_embedding(text_hash)
        
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
        assert cache._cache_stats["hits"] == 1
        assert cache._cache_stats["stores"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiration_memory(self):
        """Test cache expiration in memory mode."""
        cache = EmbeddingCache(None, ttl=0.1)  # 100ms TTL
        await cache.initialize()
        
        text_hash = "expiry_test"
        embedding = np.array([1.0, 2.0], dtype=np.float32)
        
        # Store embedding
        await cache.cache_embedding(text_hash, embedding)
        
        # Should be available immediately
        result = await cache.get_cached_embedding(text_hash)
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired now
        result = await cache.get_cached_embedding(text_hash)
        assert result is None
        assert cache._cache_stats["evictions"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, mock_redis_batching):
        """Test cache statistics collection."""
        cache = EmbeddingCache(mock_redis_batching)
        await cache.initialize()
        
        # Simulate some cache operations
        cache._cache_stats["hits"] = 15
        cache._cache_stats["misses"] = 5
        cache._cache_stats["stores"] = 10
        cache._cache_stats["evictions"] = 2
        
        # Mock Redis scan for cache size
        mock_redis_batching.scan.return_value = (0, ["embedding:hash1", "embedding:hash2"])
        
        stats = await cache.get_cache_stats()
        
        assert stats["hits"] == 15
        assert stats["misses"] == 5
        assert stats["total_requests"] == 20
        assert stats["hit_rate"] == 0.75
        assert stats["miss_rate"] == 0.25
        assert stats["cache_backend"] == "redis"
        assert stats["cache_size"] == 2
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, mock_redis_batching):
        """Test cache invalidation functionality."""
        cache = EmbeddingCache(mock_redis_batching)
        await cache.initialize()
        
        text_hash = "invalidate_test"
        
        await cache.invalidate_cache(text_hash)
        
        # Verify Redis delete was called
        mock_redis_batching.delete.assert_called_once_with(f"embedding:{text_hash}")
    
    @pytest.mark.asyncio
    async def test_cache_clearing(self, mock_redis_batching):
        """Test clearing entire cache."""
        cache = EmbeddingCache(mock_redis_batching)
        await cache.initialize()
        
        # Mock Redis scan to return some keys
        mock_redis_batching.scan.side_effect = [
            (0, ["embedding:hash1", "embedding:hash2"]),  # First scan
        ]
        mock_redis_batching.delete.return_value = 2
        
        await cache.clear_cache()
        
        # Verify Redis scan and delete were called
        mock_redis_batching.scan.assert_called()
        mock_redis_batching.delete.assert_called_once_with("embedding:hash1", "embedding:hash2")


@pytest.mark.unit
@pytest.mark.batching
class TestBatchProcessor:
    """Unit tests for batch processor component."""
    
    @pytest.fixture
    def mock_ollama_config(self):
        """Mock Ollama configuration for testing."""
        return {
            "base_url": "http://localhost:11434",
            "model": "bge-m3:latest",
            "embedding_dim": 128,
            "timeout": 30.0,
            "max_batch_size": 32,
            "min_batch_size": 1
        }
    
    def test_processor_initialization(self, mock_ollama_config):
        """Test batch processor initialization."""
        processor = BatchProcessor(mock_ollama_config)
        
        assert processor.model_name == "bge-m3:latest"
        assert processor.embedding_dim == 128
        assert processor.timeout == 30.0
        assert processor.optimal_batch_size == 16  # Optimized for bge-m3
    
    def test_batch_size_optimization(self):
        """Test batch size optimization for different models."""
        # Test bge-m3 optimization
        config = {"model": "bge-m3:latest", "max_batch_size": 32, "min_batch_size": 1}
        processor = BatchProcessor(config)
        assert processor.optimal_batch_size == 16
        
        # Test all-minilm optimization
        config["model"] = "all-minilm:latest"
        processor = BatchProcessor(config)
        assert processor.optimal_batch_size == 32
        
        # Test unknown model (conservative default)
        config["model"] = "unknown-model"
        processor = BatchProcessor(config)
        assert processor.optimal_batch_size == 16
    
    def test_batch_splitting(self, mock_ollama_config):
        """Test splitting large batches into optimal sub-batches."""
        processor = BatchProcessor(mock_ollama_config)
        processor.optimal_batch_size = 4
        
        # Test splitting large batch
        texts = [f"text_{i}" for i in range(10)]
        sub_batches = processor._split_into_sub_batches(texts)
        
        assert len(sub_batches) == 3  # 4, 4, 2
        assert len(sub_batches[0]) == 4
        assert len(sub_batches[1]) == 4
        assert len(sub_batches[2]) == 2
        
        # Test batch smaller than optimal size
        small_texts = ["text1", "text2"]
        sub_batches = processor._split_into_sub_batches(small_texts)
        
        assert len(sub_batches) == 1
        assert len(sub_batches[0]) == 2
    
    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, mock_ollama_config, mock_ollama_session):
        """Test single embedding generation."""
        processor = BatchProcessor(mock_ollama_config)
        processor._session = mock_ollama_session
        
        # Mock the batch processor process_batch method instead
        async def mock_process_batch(texts):
            return [np.array([0.1, 0.2, 0.3, 0.4] * 32, dtype=np.float32) for _ in texts]
        
        processor.process_batch = mock_process_batch
        
        embedding = await processor._generate_single_embedding("test text")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 128  # 32 * 4 from mock response
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_ollama_config):
        """Test concurrent request processing."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Mock single embedding generation
        embedding_counter = 0
        async def mock_generate_single(text):
            nonlocal embedding_counter
            embedding_counter += 1
            # Return different embeddings based on counter
            return np.array([embedding_counter] * 128, dtype=np.float32)
        
        processor._generate_single_embedding = mock_generate_single
        
        texts = ["text1", "text2", "text3"]
        embeddings = await processor._process_concurrent_requests(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (128,) for emb in embeddings)
        
        # Verify different embeddings were generated
        assert not np.array_equal(embeddings[0], embeddings[1])
    
    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_processing(self, mock_ollama_config):
        """Test error handling in concurrent processing."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Mock function that fails for some texts
        async def mock_generate_with_failures(text):
            if "fail" in text:
                raise RuntimeError("Simulated failure")
            return np.array([1.0] * 128, dtype=np.float32)
        
        processor._generate_single_embedding = mock_generate_with_failures
        
        texts = ["success1", "fail_text", "success2"]
        embeddings = await processor._process_concurrent_requests(texts)
        
        assert len(embeddings) == 3
        
        # Failed embedding should be zero array
        assert np.allclose(embeddings[1], np.zeros(128))
        
        # Successful embeddings should be non-zero
        assert not np.allclose(embeddings[0], np.zeros(128))
        assert not np.allclose(embeddings[2], np.zeros(128))
    
    @pytest.mark.asyncio
    async def test_http_session_management(self, mock_ollama_config):
        """Test HTTP session initialization and cleanup."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Initially no session
        assert processor._session is None
        
        # Initialize session
        await processor.initialize()
        assert processor._session is not None
        assert processor._connector is not None
        
        # Cleanup session
        await processor.cleanup()
        assert processor._session is None
        assert processor._connector is None
    
    def test_metrics_collection(self, mock_ollama_config):
        """Test performance metrics collection."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Manually set metrics for testing
        processor._metrics = {
            "batches_processed": 3,
            "total_texts": 18,
            "failed_batches": 1,
            "total_processing_time": 4.5,
            "processing_times": [2.5, 1.2, 0.8]
        }
        
        metrics = processor.get_metrics()
        
        assert metrics["batches_processed"] == 3
        assert metrics["total_texts"] == 18
        assert metrics["failed_batches"] == 1
        assert abs(metrics["success_rate"] - 2/3) < 0.01
        assert abs(metrics["avg_batch_size"] - 6.0) < 0.01  # 18 texts / 3 batches
        assert abs(metrics["avg_processing_time"] - 1.5) < 0.01  # Total time / batches
    
    def test_metrics_reset(self, mock_ollama_config):
        """Test metrics reset functionality."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Set some metrics
        processor._update_metrics(5, 1.0, success=True)
        
        # Verify metrics exist
        metrics = processor.get_metrics()
        assert metrics["total_texts"] > 0
        
        # Reset metrics
        processor.reset_metrics()
        
        # Verify metrics are reset
        metrics = processor.get_metrics()
        assert metrics["total_texts"] == 0
        assert metrics["batches_processed"] == 0


@pytest.mark.unit
@pytest.mark.batching
class TestErrorHandler:
    """Unit tests for error handler component."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = BatchProcessingErrorHandler(max_retries=5, backoff_multiplier=1.5)
        
        assert handler.max_retries == 5
        assert handler.backoff_multiplier == 1.5
        assert handler._circuit_breaker["state"] == "closed"
        assert handler._circuit_breaker["failure_count"] == 0
    
    def test_error_categorization(self):
        """Test error categorization logic."""
        handler = BatchProcessingErrorHandler()
        
        # Test timeout errors
        assert handler._categorize_error("Connection timeout") == "timeout"
        assert handler._categorize_error("Request timed out") == "timeout"
        assert handler._categorize_error("Deadline exceeded") == "timeout"
        
        # Test connection errors
        assert handler._categorize_error("Connection refused") == "connection"
        assert handler._categorize_error("Network unreachable") == "connection"
        assert handler._categorize_error("Connection failed") == "connection"
        
        # Test rate limit errors
        assert handler._categorize_error("Rate limit exceeded") == "rate_limit"
        assert handler._categorize_error("Too many requests") == "rate_limit"
        assert handler._categorize_error("Quota exceeded") == "rate_limit"
        
        # Test transient errors
        assert handler._categorize_error("HTTP 502 error") == "transient"
        assert handler._categorize_error("503 Service Unavailable") == "transient"
        assert handler._categorize_error("Server error occurred") == "transient"
        
        # Test permanent errors
        assert handler._categorize_error("Invalid model specified") == "permanent"
        assert handler._categorize_error("Bad request format") == "permanent"
        assert handler._categorize_error("400 Bad Request") == "permanent"
        
        # Test unknown errors (default to transient)
        assert handler._categorize_error("Unknown error occurred") == "transient"
    
    def test_backoff_calculation(self):
        """Test exponential backoff calculation."""
        handler = BatchProcessingErrorHandler(backoff_multiplier=2.0)
        
        # Test increasing backoff
        delay0 = handler._calculate_backoff_delay(0)
        delay1 = handler._calculate_backoff_delay(1)
        delay2 = handler._calculate_backoff_delay(2)
        delay3 = handler._calculate_backoff_delay(3)
        
        # Should increase exponentially (with jitter)
        assert 0.5 <= delay0 <= 1.5  # Base delay with jitter
        assert 1.0 <= delay1 <= 3.0  # 2x base with jitter
        assert 2.0 <= delay2 <= 6.0  # 4x base with jitter
        assert 4.0 <= delay3 <= 12.0  # 8x base with jitter
        
        # Test maximum delay cap
        delay_large = handler._calculate_backoff_delay(10)
        assert delay_large <= 60.0  # Should be capped at 60 seconds
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        handler = BatchProcessingErrorHandler()
        handler._circuit_breaker["failure_threshold"] = 3
        
        # Initial state should be closed
        assert handler._circuit_breaker["state"] == "closed"
        assert await handler._can_retry() is True
        
        # First failure
        await handler._update_circuit_breaker("test error 1")
        assert handler._circuit_breaker["state"] == "closed"
        assert handler._circuit_breaker["failure_count"] == 1
        
        # Second failure
        await handler._update_circuit_breaker("test error 2")
        assert handler._circuit_breaker["state"] == "closed"
        assert handler._circuit_breaker["failure_count"] == 2
        
        # Third failure - should open circuit
        await handler._update_circuit_breaker("test error 3")
        assert handler._circuit_breaker["state"] == "open"
        assert handler._circuit_breaker["failure_count"] == 3
        assert await handler._can_retry() is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery to half-open state."""
        handler = BatchProcessingErrorHandler()
        handler._circuit_breaker["failure_threshold"] = 2
        handler._circuit_breaker["recovery_timeout"] = 0.1  # 100ms for testing
        
        # Force circuit to open
        await handler._update_circuit_breaker("error 1")
        await handler._update_circuit_breaker("error 2")
        assert handler._circuit_breaker["state"] == "open"
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Should transition to half-open on next retry check
        assert await handler._can_retry() is True
        assert handler._circuit_breaker["state"] == "half-open"
        
        # Failure in half-open should go back to open
        await handler._update_circuit_breaker("error in half-open")
        assert handler._circuit_breaker["state"] == "open"
    
    @pytest.mark.asyncio
    async def test_request_retry_logic(self):
        """Test individual request retry logic."""
        handler = BatchProcessingErrorHandler(max_retries=3)
        
        # Test retry allowance
        request = {"request_id": "test", "retry_count": 0}
        should_retry = await handler.can_retry_request(request, "test error")
        assert should_retry is True
        
        # Test retry count limit
        request["retry_count"] = 3  # At max retries
        should_retry = await handler.can_retry_request(request, "test error")
        assert should_retry is False
        
        # Test circuit breaker blocking
        handler._circuit_breaker["state"] = "open"
        request["retry_count"] = 0
        should_retry = await handler.can_retry_request(request, "test error")
        assert should_retry is False
    
    @pytest.mark.asyncio
    async def test_batch_failure_handling(self):
        """Test batch failure handling."""
        handler = BatchProcessingErrorHandler()
        
        requests = [
            {"request_id": "req1", "text": "text1"},
            {"request_id": "req2", "text": "text2"}
        ]
        
        # This should update error statistics
        await handler.handle_batch_failure(requests, "Test batch failure", {"context": "test"})
        
        stats = handler.get_error_stats()
        assert stats["total_errors"] == 1
        assert "transient" in stats["error_types"]  # Default categorization
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        handler = BatchProcessingErrorHandler()
        
        # Test normal timeout
        await handler.handle_timeout("batch_123", 15.0)
        
        # Test long timeout that should trigger circuit breaker
        await handler.handle_timeout("batch_456", 35.0)
        assert handler._circuit_breaker["state"] == "open"
    
    @pytest.mark.asyncio
    async def test_ollama_unavailable_handling(self):
        """Test Ollama service unavailable handling."""
        handler = BatchProcessingErrorHandler()
        
        await handler.handle_ollama_unavailable()
        
        # Should immediately open circuit breaker
        assert handler._circuit_breaker["state"] == "open"
        assert handler._circuit_breaker["failure_count"] >= handler._circuit_breaker["failure_threshold"]
        
        # Should update error statistics
        stats = handler.get_error_stats()
        assert stats["total_errors"] == 1
        assert "service_unavailable" in stats["error_types"]
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        handler = BatchProcessingErrorHandler()
        
        # Update statistics manually for testing
        handler._update_error_stats("timeout error", 5)
        handler._update_error_stats("connection error", 3)
        handler._update_error_stats("timeout error", 2)
        
        stats = handler.get_error_stats()
        
        assert stats["total_errors"] == 3
        assert stats["error_types"]["timeout"] == 2
        assert stats["error_types"]["connection"] == 1
    
    def test_manual_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        handler = BatchProcessingErrorHandler()
        
        # Force circuit to open
        handler._circuit_breaker["state"] = "open"
        handler._circuit_breaker["failure_count"] = 10
        
        # Reset circuit breaker
        handler.reset_circuit_breaker()
        
        assert handler._circuit_breaker["state"] == "closed"
        assert handler._circuit_breaker["failure_count"] == 0
        assert handler._circuit_breaker["last_failure_time"] == 0
    
    def test_statistics_reset(self):
        """Test error statistics reset."""
        handler = BatchProcessingErrorHandler()
        
        # Set some statistics
        handler._error_stats["total_errors"] = 10
        handler._error_stats["retry_attempts"] = 15
        handler._error_stats["error_types"]["timeout"] = 5
        
        # Reset statistics
        handler.reset_stats()
        
        assert handler._error_stats["total_errors"] == 0
        assert handler._error_stats["retry_attempts"] == 0
        assert handler._error_stats["error_types"] == {}


@pytest.mark.unit
@pytest.mark.batching
class TestBatchingConfig:
    """Unit tests for batching configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults."""
        config = BatchingConfig()
        
        assert config.batch_size > 0
        assert config.max_batch_size >= config.batch_size
        assert config.cache_ttl > 0
        assert config.max_retries > 0
        assert config.ollama_base_url.startswith("http")
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = BatchingConfig(
            batch_size=16,
            max_batch_size=32,
            min_batch_size=4,
            cache_ttl=3600
        )
        assert config.batch_size == 16
        
        # Test batch size adjustment
        config = BatchingConfig(
            batch_size=50,  # Too large
            max_batch_size=32
        )
        assert config.batch_size == 32  # Should be adjusted
        
        # Test minimum batch size enforcement
        config = BatchingConfig(
            batch_size=1,  # Too small
            min_batch_size=4
        )
        assert config.batch_size == 4  # Should be adjusted
    
    def test_model_specific_optimization(self):
        """Test model-specific optimization."""
        base_config = BatchingConfig()
        
        # Test bge-m3 optimization
        optimized = base_config.get_optimized_config_for_model("bge-m3:latest")
        assert optimized.batch_size == 16
        assert hasattr(optimized, 'embedding_dim')
        
        # Test all-minilm optimization
        optimized = base_config.get_optimized_config_for_model("all-minilm:latest")
        assert optimized.batch_size == 32
        assert hasattr(optimized, 'embedding_dim')
        
        # Test unknown model (conservative defaults)
        optimized = base_config.get_optimized_config_for_model("unknown-model")
        assert optimized.batch_size >= 1  # Conservative but reasonable
    
    def test_config_conversion_methods(self):
        """Test configuration conversion methods."""
        config = BatchingConfig(
            ollama_base_url="http://test:11434",
            ollama_model="test-model",
            embedding_dim=256,
            redis_host="redis-host",
            redis_port=6380,
            redis_db=3
        )
        
        # Test Ollama config conversion
        ollama_config = config.to_ollama_config()
        assert ollama_config["base_url"] == "http://test:11434"
        assert ollama_config["model"] == "test-model"
        assert ollama_config["embedding_dim"] == 256
        
        # Test Redis config conversion
        redis_config = config.to_redis_config()
        assert redis_config["host"] == "redis-host"
        assert redis_config["port"] == 6380
        assert redis_config["db"] == 3
        assert redis_config["decode_responses"] is True
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "batch_size": 24,
            "cache_ttl": 7200,
            "ollama_model": "custom-model",
            "unknown_field": "should_be_ignored"  # Should be filtered out
        }
        
        config = BatchingConfig.from_dict(config_dict)
        
        assert config.batch_size == 24
        assert config.cache_ttl == 7200
        assert config.ollama_model == "custom-model"
        # unknown_field should be ignored
    
    @patch.dict('os.environ', {
        'OLLAMA_EMBEDDING_BATCH_SIZE': '20',
        'EMBEDDING_CACHE_TTL': '1800',
        'REDIS_EMBEDDING_DB': '5'
    })
    def test_config_from_environment(self):
        """Test creating configuration from environment variables."""
        config = BatchingConfig.from_env()
        
        assert config.batch_size == 20
        assert config.cache_ttl == 1800
        assert config.redis_db == 5


if __name__ == "__main__":
    # Run the unit tests
    pytest.main([__file__, "-v", "-m", "unit and batching"])