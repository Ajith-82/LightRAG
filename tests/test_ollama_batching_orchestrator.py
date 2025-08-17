"""
Comprehensive tests for Ollama Batching Orchestrator.

Tests all components of the batching system including orchestrator,
queue management, caching, batch processing, and error handling.
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.orchestrator import (
    OllamaBatchingOrchestrator,
    RequestQueue,
    EmbeddingCache,
    BatchProcessor,
    BatchProcessingErrorHandler
)
from lightrag.config.batch_config import BatchingConfig
from lightrag.pipeline.enhanced_embedding import EnhancedEmbeddingPipeline
from lightrag.pipeline.batch_integration import LightRAGBatchIntegration


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.llen.return_value = 0
    mock_client.lpush.return_value = 1
    mock_client.lrange.return_value = []
    mock_client.ltrim.return_value = True
    mock_client.delete.return_value = 1
    mock_client.setex.return_value = True
    mock_client.get.return_value = None
    mock_client.scan.return_value = (0, [])
    mock_client.pipeline.return_value = mock_client
    mock_client.execute.return_value = [[], True]
    mock_client.select.return_value = True
    return mock_client


@pytest.fixture
def test_config():
    """Test configuration with optimized settings."""
    return BatchingConfig(
        batch_size=4,
        max_batch_size=8,
        batch_timeout=5000,  # 5 seconds for testing
        processing_interval=0.01,  # 10ms for fast testing
        cache_ttl=60,
        max_retries=2,
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        redis_db=15  # Use test database
    )


class TestRequestQueue:
    """Test request queue functionality."""
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self, mock_redis):
        """Test queue initialization."""
        queue = RequestQueue(mock_redis)
        await queue.initialize()
        
        assert queue._initialized is True
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enqueue_request(self, mock_redis):
        """Test request enqueueing."""
        queue = RequestQueue(mock_redis)
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
        
        # Verify Redis lpush was called
        mock_redis.lpush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dequeue_batch(self, mock_redis):
        """Test batch dequeuing."""
        queue = RequestQueue(mock_redis)
        await queue.initialize()
        
        # Mock Redis pipeline responses
        mock_redis.execute.return_value = [
            ['{"request_id": "test-1", "text": "text1"}'],  # lrange result
            True  # ltrim result
        ]
        
        batch = await queue.dequeue_batch(2)
        
        assert len(batch) == 1
        assert batch[0]["request_id"] == "test-1"
    
    @pytest.mark.asyncio
    async def test_in_memory_fallback(self):
        """Test in-memory queue fallback when Redis unavailable."""
        queue = RequestQueue(None)  # No Redis client
        await queue.initialize()
        
        request_data = {
            "request_id": "test-123",
            "text": "test text",
            "priority": 1
        }
        
        await queue.enqueue_request(request_data)
        batch = await queue.dequeue_batch(5)
        
        assert len(batch) == 1
        assert batch[0]["request_id"] == "test-123"


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, mock_redis):
        """Test cache initialization."""
        cache = EmbeddingCache(mock_redis, db=2, ttl=3600)
        await cache.initialize()
        
        assert cache._initialized is True
        mock_redis.select.assert_called_once_with(2)
    
    @pytest.mark.asyncio
    async def test_text_hashing(self, mock_redis):
        """Test consistent text hashing."""
        cache = EmbeddingCache(mock_redis)
        await cache.initialize()
        
        text = "Test text for hashing"
        hash1 = await cache.get_text_hash(text)
        hash2 = await cache.get_text_hash(text)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, mock_redis):
        """Test cache store and retrieve operations."""
        cache = EmbeddingCache(mock_redis, ttl=60)
        await cache.initialize()
        
        text_hash = "test_hash_123"
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        # Test cache miss
        result = await cache.get_cached_embedding(text_hash)
        assert result is None
        
        # Test cache store
        await cache.cache_embedding(text_hash, embedding)
        mock_redis.setex.assert_called_once()
        
        # Test cache hit (mock Redis to return the embedding)
        import pickle
        mock_redis.get.return_value = pickle.dumps(embedding)
        result = await cache.get_cached_embedding(text_hash)
        
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, mock_redis):
        """Test cache statistics tracking."""
        cache = EmbeddingCache(mock_redis)
        await cache.initialize()
        
        # Simulate cache operations
        cache._cache_stats["hits"] = 10
        cache._cache_stats["misses"] = 5
        cache._cache_stats["stores"] = 8
        
        stats = await cache.get_cache_stats()
        
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["total_requests"] == 15
        assert stats["hit_rate"] == 10/15


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    @pytest.fixture
    def mock_ollama_config(self):
        """Mock Ollama configuration."""
        return {
            "base_url": "http://localhost:11434",
            "model": "test-model",
            "embedding_dim": 128,
            "timeout": 30.0
        }
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, mock_ollama_config):
        """Test batch processor initialization."""
        processor = BatchProcessor(mock_ollama_config)
        
        assert processor.model_name == "test-model"
        assert processor.embedding_dim == 128
        assert processor.optimal_batch_size == 16  # Default for unknown models
    
    @pytest.mark.asyncio
    async def test_batch_splitting(self, mock_ollama_config):
        """Test large batch splitting into sub-batches."""
        processor = BatchProcessor(mock_ollama_config)
        processor.optimal_batch_size = 4
        
        texts = [f"text_{i}" for i in range(10)]
        sub_batches = processor._split_into_sub_batches(texts)
        
        assert len(sub_batches) == 3  # 4, 4, 2
        assert len(sub_batches[0]) == 4
        assert len(sub_batches[1]) == 4
        assert len(sub_batches[2]) == 2
    
    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, mock_ollama_config):
        """Test single embedding generation with mocked Ollama."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Mock HTTP session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4]
        }
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        processor._session = mock_session
        
        embedding = await processor._generate_single_embedding("test text")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (4,)
        np.testing.assert_array_almost_equal(embedding, [0.1, 0.2, 0.3, 0.4])
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_ollama_config):
        """Test concurrent request processing."""
        processor = BatchProcessor(mock_ollama_config)
        
        # Mock single embedding generation
        async def mock_generate_single(text):
            # Simulate different embeddings based on text
            seed = hash(text) % 100
            return np.array([seed/100, (seed+1)/100, (seed+2)/100], dtype=np.float32)
        
        processor._generate_single_embedding = mock_generate_single
        
        texts = ["text1", "text2", "text3"]
        embeddings = await processor._process_concurrent_requests(texts)
        
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (3,) for emb in embeddings)


class TestErrorHandler:
    """Test error handling functionality."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = BatchProcessingErrorHandler(max_retries=5, backoff_multiplier=1.5)
        
        assert handler.max_retries == 5
        assert handler.backoff_multiplier == 1.5
        assert handler._circuit_breaker["state"] == "closed"
    
    def test_error_categorization(self):
        """Test error categorization logic."""
        handler = BatchProcessingErrorHandler()
        
        assert handler._categorize_error("Connection timeout") == "timeout"
        assert handler._categorize_error("Connection refused") == "connection"
        assert handler._categorize_error("Rate limit exceeded") == "rate_limit"
        assert handler._categorize_error("HTTP 502 error") == "transient"
        assert handler._categorize_error("Invalid model") == "permanent"
        assert handler._categorize_error("Unknown error") == "transient"
    
    def test_backoff_calculation(self):
        """Test exponential backoff calculation."""
        handler = BatchProcessingErrorHandler(backoff_multiplier=2.0)
        
        delay1 = handler._calculate_backoff_delay(0)
        delay2 = handler._calculate_backoff_delay(1)
        delay3 = handler._calculate_backoff_delay(2)
        
        # Should increase exponentially (with jitter)
        assert 0.5 <= delay1 <= 1.5  # Base delay with jitter
        assert 1.0 <= delay2 <= 3.0  # 2x base with jitter
        assert 2.0 <= delay3 <= 6.0  # 4x base with jitter
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_transitions(self):
        """Test circuit breaker state transitions."""
        handler = BatchProcessingErrorHandler()
        handler._circuit_breaker["failure_threshold"] = 2
        
        # Initial state should be closed
        assert handler._circuit_breaker["state"] == "closed"
        
        # First failure
        await handler._update_circuit_breaker("test error")
        assert handler._circuit_breaker["state"] == "closed"
        assert handler._circuit_breaker["failure_count"] == 1
        
        # Second failure - should open circuit
        await handler._update_circuit_breaker("test error")
        assert handler._circuit_breaker["state"] == "open"
        assert handler._circuit_breaker["failure_count"] == 2
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test request retry logic."""
        handler = BatchProcessingErrorHandler(max_retries=3)
        
        request = {"request_id": "test", "retry_count": 0}
        
        # Should allow retry for first failure
        should_retry = await handler.handle_request_retry(request, "test error")
        assert should_retry is True
        
        # Should not retry after max retries
        request["retry_count"] = 3
        should_retry = await handler.handle_request_retry(request, "test error")
        assert should_retry is False


class TestOllamaBatchingOrchestrator:
    """Test the main orchestrator integration."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_redis, test_config):
        """Test orchestrator initialization."""
        orchestrator = OllamaBatchingOrchestrator(
            redis_client=mock_redis,
            batch_size=test_config.batch_size,
            timeout=test_config.batch_timeout,
            ollama_config=test_config.to_ollama_config()
        )
        
        assert orchestrator.batch_size == test_config.batch_size
        assert not orchestrator._running
        assert len(orchestrator._active_requests) == 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self, mock_redis, test_config):
        """Test orchestrator start and stop operations."""
        orchestrator = OllamaBatchingOrchestrator(
            redis_client=mock_redis,
            batch_size=test_config.batch_size,
            ollama_config=test_config.to_ollama_config()
        )
        
        # Test start
        await orchestrator.start()
        assert orchestrator._running is True
        assert orchestrator._processing_task is not None
        
        # Test stop
        await orchestrator.stop()
        assert orchestrator._running is False
    
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, mock_redis, test_config):
        """Test cache hit scenario."""
        orchestrator = OllamaBatchingOrchestrator(
            redis_client=mock_redis,
            batch_size=test_config.batch_size,
            ollama_config=test_config.to_ollama_config()
        )
        
        # Mock cache hit
        cached_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        orchestrator.embedding_cache.get_cached_embedding = AsyncMock(return_value=cached_embedding)
        
        await orchestrator.start()
        
        # Queue request
        request_id = await orchestrator.queue_embedding_request("test text")
        
        # Should get cached result immediately
        result = await orchestrator.get_embedding_result(request_id, timeout=1.0)
        
        np.testing.assert_array_equal(result, cached_embedding)
        
        # Verify metrics
        metrics = orchestrator.get_metrics()
        assert metrics["cache_hit_rate"] == 1.0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_redis, test_config):
        """Test metrics collection."""
        orchestrator = OllamaBatchingOrchestrator(
            redis_client=mock_redis,
            batch_size=test_config.batch_size,
            ollama_config=test_config.to_ollama_config()
        )
        
        # Simulate some operations
        orchestrator._metrics["total_requests"] = 100
        orchestrator._metrics["cache_hits"] = 75
        orchestrator._metrics["cache_misses"] = 25
        orchestrator._metrics["batches_processed"] = 10
        orchestrator._metrics["total_processing_time"] = 50.0
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["total_requests"] == 100
        assert metrics["cache_hit_rate"] == 0.75
        assert metrics["cache_miss_rate"] == 0.25
        assert metrics["batches_processed"] == 10
        assert metrics["average_batch_time"] == 5.0


class TestEnhancedEmbeddingPipeline:
    """Test enhanced embedding pipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_redis, test_config):
        """Test pipeline initialization."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis,
            config=test_config
        )
        
        assert pipeline.config == test_config
        assert not pipeline._started
    
    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, mock_redis, test_config):
        """Test single embedding generation."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis,
            config=test_config
        )
        
        # Mock orchestrator
        mock_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        pipeline.orchestrator.queue_embedding_request = AsyncMock(return_value="request-123")
        pipeline.orchestrator.get_embedding_result = AsyncMock(return_value=mock_embedding)
        pipeline.orchestrator.start = AsyncMock()
        
        embedding = await pipeline.generate_single_embedding("test text")
        
        np.testing.assert_array_equal(embedding, mock_embedding)
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_redis, test_config):
        """Test batch embedding generation."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis,
            config=test_config
        )
        
        # Mock orchestrator
        mock_embeddings = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32)
        ]
        pipeline.orchestrator.process_embeddings_batch = AsyncMock(return_value=mock_embeddings)
        pipeline.orchestrator.start = AsyncMock()
        
        texts = ["text1", "text2"]
        embeddings = await pipeline.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        np.testing.assert_array_equal(embeddings[0], mock_embeddings[0])
        np.testing.assert_array_equal(embeddings[1], mock_embeddings[1])
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, mock_redis, test_config):
        """Test fallback to original function when orchestrator fails."""
        def mock_fallback_func(texts):
            return [np.array([0.5, 0.6], dtype=np.float32) for _ in texts]
        
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis,
            config=test_config,
            fallback_func=mock_fallback_func
        )
        
        # Mock orchestrator to fail
        pipeline.orchestrator.process_embeddings_batch = AsyncMock(side_effect=RuntimeError("Orchestrator failed"))
        pipeline.orchestrator.start = AsyncMock()
        
        texts = ["text1"]
        embeddings = await pipeline.generate_embeddings(texts)
        
        assert len(embeddings) == 1
        np.testing.assert_array_equal(embeddings[0], [0.5, 0.6])
        
        # Check fallback metrics
        metrics = pipeline.get_performance_metrics()
        assert metrics["pipeline_metrics"]["fallback_requests"] == 1


class TestLightRAGBatchIntegration:
    """Test LightRAG integration functionality."""
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, test_config):
        """Test integration initialization."""
        integration = LightRAGBatchIntegration(
            config=test_config,
            redis_url="redis://localhost:6379/15",
            enable_batching=False  # Disable for testing without Redis
        )
        
        assert integration.config == test_config
        assert not integration.enable_batching
        assert not integration._initialized
    
    @pytest.mark.asyncio
    async def test_disabled_batching(self, test_config):
        """Test integration with batching disabled."""
        integration = LightRAGBatchIntegration(
            config=test_config,
            enable_batching=False
        )
        
        # Mock original function
        def original_func(texts):
            return [np.array([1.0, 2.0], dtype=np.float32) for _ in texts]
        
        # Wrap function (should return original when disabled)
        wrapped_func = integration.wrap_embedding_function(original_func)
        
        assert wrapped_func == original_func
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_config):
        """Test integration health check."""
        integration = LightRAGBatchIntegration(
            config=test_config,
            enable_batching=False
        )
        
        health = await integration.health_check()
        
        assert health["integration_status"] == "disabled"
        assert "redis_status" in health
        assert "pipeline_status" in health


class TestPerformanceOptimizations:
    """Test performance optimizations and benchmarks."""
    
    @pytest.mark.asyncio
    async def test_batching_performance_simulation(self, mock_redis, test_config):
        """Simulate batching performance improvements."""
        # This test simulates the expected 50% performance improvement
        
        orchestrator = OllamaBatchingOrchestrator(
            redis_client=mock_redis,
            batch_size=8,
            ollama_config=test_config.to_ollama_config()
        )
        
        # Mock batch processor with realistic timing
        async def mock_process_batch(texts):
            # Simulate batch processing time (more efficient than individual)
            await asyncio.sleep(0.1 * len(texts) / 8)  # 0.1s per 8 texts
            return [np.random.rand(128).astype(np.float32) for _ in texts]
        
        orchestrator.batch_processor.process_batch = mock_process_batch
        
        # Test batch processing time
        texts = [f"text_{i}" for i in range(16)]  # 2 batches
        
        start_time = time.time()
        await orchestrator.start()
        
        # Process all texts through orchestrator
        embeddings = await orchestrator.process_embeddings_batch(texts)
        
        await orchestrator.stop()
        batch_time = time.time() - start_time
        
        assert len(embeddings) == 16
        
        # Simulate individual processing time (baseline)
        individual_time = 0.1 * 16  # 0.1s per text individually
        
        # Batch processing should be faster
        performance_improvement = (individual_time - batch_time) / individual_time
        
        # Should achieve significant improvement (this is a simulation)
        assert performance_improvement > 0.3, f"Performance improvement only {performance_improvement:.2%}"
    
    @pytest.mark.asyncio
    async def test_cache_efficiency(self, mock_redis, test_config):
        """Test cache efficiency for repeated requests."""
        orchestrator = OllamaBatchingOrchestrator(
            redis_client=mock_redis,
            batch_size=4,
            ollama_config=test_config.to_ollama_config()
        )
        
        # Mock cache with some pre-existing embeddings
        cached_embeddings = {
            "text_1": np.array([0.1, 0.2], dtype=np.float32),
            "text_2": np.array([0.3, 0.4], dtype=np.float32)
        }
        
        async def mock_get_cached(text_hash):
            # Simulate cache hits for some texts
            for text, embedding in cached_embeddings.items():
                if text in text_hash:
                    return embedding
            return None
        
        orchestrator.embedding_cache.get_cached_embedding = mock_get_cached
        orchestrator.embedding_cache.get_text_hash = AsyncMock(side_effect=lambda x: f"hash_{x}")
        
        await orchestrator.start()
        
        # Request embeddings (some cached, some not)
        texts = ["text_1", "text_2", "text_3", "text_4"]
        request_ids = []
        
        for text in texts:
            request_id = await orchestrator.queue_embedding_request(text)
            request_ids.append(request_id)
        
        # Check metrics
        metrics = orchestrator.get_metrics()
        expected_cache_hits = 2  # text_1 and text_2 should be cache hits
        
        assert metrics["cache_hits"] >= expected_cache_hits
        
        await orchestrator.stop()


# Integration test
@pytest.mark.asyncio
async def test_end_to_end_integration():
    """Test complete end-to-end integration."""
    config = BatchingConfig(
        batch_size=2,
        processing_interval=0.01,
        cache_ttl=60,
        redis_db=15
    )
    
    # Create integration without Redis for testing
    integration = LightRAGBatchIntegration(
        config=config,
        enable_batching=False  # Disable for testing
    )
    
    # Test configuration
    config_summary = integration.get_configuration_summary()
    assert config_summary["batching_enabled"] is False
    assert config_summary["config"]["batch_size"] == 2
    
    # Test health check
    health = await integration.health_check()
    assert health["integration_status"] == "disabled"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])