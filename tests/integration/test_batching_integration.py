"""
Integration tests for Ollama Batching Orchestrator with LightRAG.

Tests the complete integration of batching functionality with 
existing LightRAG workflows and performance improvements.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch
import numpy as np

from lightrag.pipeline.batch_integration import (
    LightRAGBatchIntegration,
    enable_lightrag_batching,
    is_batching_available,
    get_recommended_config
)
from lightrag.pipeline.enhanced_embedding import EnhancedEmbeddingPipeline
from lightrag.orchestrator import OllamaBatchingOrchestrator


@pytest.mark.integration
@pytest.mark.batching
class TestLightRAGBatchIntegration:
    """Test LightRAG integration with batching orchestrator."""
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, batching_config, mock_redis_batching):
        """Test integration initialization with mocked components."""
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=True
        )
        
        # Mock Redis creation to use our mock
        with patch('lightrag.pipeline.batch_integration.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_batching
            
            await integration.initialize()
            
            assert integration._initialized is True
            assert integration.enable_batching is True
    
    @pytest.mark.asyncio
    async def test_integration_disabled_batching(self, batching_config):
        """Test integration with batching disabled."""
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=False
        )
        
        await integration.initialize()
        
        assert integration._initialized is True
        assert integration.enable_batching is False
    
    @pytest.mark.asyncio
    async def test_embedding_function_wrapping(self, batching_config, mock_redis_batching):
        """Test wrapping of embedding functions."""
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=False  # Disable for function wrapping test
        )
        
        # Original function
        def original_embedding_func(texts):
            return [np.random.rand(128).astype(np.float32) for _ in texts]
        
        # Wrap function
        wrapped_func = integration.wrap_embedding_function(original_embedding_func)
        
        # Should return original function when batching disabled
        assert wrapped_func == original_embedding_func
    
    @pytest.mark.asyncio
    async def test_health_check_functionality(self, batching_config, mock_redis_batching):
        """Test health check functionality."""
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=True
        )
        
        # Mock Redis for health check
        with patch('lightrag.pipeline.batch_integration.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_batching
            
            await integration.initialize()
            
            # Test health check
            health = await integration.health_check()
            
            assert "integration_status" in health
            assert "redis_status" in health
            assert "pipeline_status" in health
    
    @pytest.mark.asyncio
    async def test_configuration_summary(self, batching_config):
        """Test configuration summary generation."""
        integration = LightRAGBatchIntegration(config=batching_config)
        
        summary = integration.get_configuration_summary()
        
        assert "batching_enabled" in summary
        assert "config" in summary
        assert summary["config"]["batch_size"] == batching_config.batch_size
        assert summary["config"]["model"] == batching_config.ollama_model
    
    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, batching_config, mock_redis_batching):
        """Test proper cleanup of resources."""
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=True
        )
        
        with patch('lightrag.pipeline.batch_integration.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_batching
            
            await integration.initialize()
            assert integration._initialized is True
            
            await integration.cleanup()
            assert integration._initialized is False
            
            # Verify Redis cleanup was called
            mock_redis_batching.close.assert_called_once()


@pytest.mark.integration
@pytest.mark.batching
class TestEnhancedEmbeddingPipelineIntegration:
    """Test enhanced embedding pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, batching_config, mock_redis_batching):
        """Test enhanced pipeline initialization."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        assert pipeline.config == batching_config
        assert not pipeline._started
    
    @pytest.mark.asyncio
    async def test_single_embedding_generation(self, batching_config, mock_redis_batching):
        """Test single embedding generation through pipeline."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock orchestrator methods
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        pipeline.orchestrator.queue_embedding_request = AsyncMock(return_value="test_id")
        pipeline.orchestrator.get_embedding_result = AsyncMock(return_value=mock_embedding)
        pipeline.orchestrator.start = AsyncMock()
        
        result = await pipeline.generate_single_embedding("test text")
        
        np.testing.assert_array_equal(result, mock_embedding)
        pipeline.orchestrator.queue_embedding_request.assert_called_once()
        pipeline.orchestrator.get_embedding_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, batching_config, mock_redis_batching, sample_texts):
        """Test batch embedding generation through pipeline."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock batch processing
        mock_embeddings = [np.random.rand(128).astype(np.float32) for _ in sample_texts]
        pipeline.orchestrator.process_embeddings_batch = AsyncMock(return_value=mock_embeddings)
        pipeline.orchestrator.start = AsyncMock()
        
        results = await pipeline.generate_embeddings(sample_texts)
        
        assert len(results) == len(sample_texts)
        assert all(isinstance(emb, np.ndarray) for emb in results)
        pipeline.orchestrator.process_embeddings_batch.assert_called_once_with(sample_texts)
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, batching_config, mock_redis_batching, sample_texts):
        """Test fallback to original function when orchestrator fails."""
        def fallback_func(texts):
            return [np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32) for _ in texts]
        
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config,
            fallback_func=fallback_func
        )
        
        # Mock orchestrator to fail
        pipeline.orchestrator.process_embeddings_batch = AsyncMock(
            side_effect=RuntimeError("Orchestrator failed")
        )
        pipeline.orchestrator.start = AsyncMock()
        
        results = await pipeline.generate_embeddings(sample_texts[:2])
        
        assert len(results) == 2
        for result in results:
            np.testing.assert_array_almost_equal(result, [0.5, 0.6, 0.7, 0.8], decimal=6)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, batching_config, mock_redis_batching):
        """Test performance metrics collection."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock orchestrator metrics
        mock_orchestrator_metrics = {
            "total_requests": 10,
            "cache_hit_rate": 0.75,
            "batches_processed": 3,
            "average_batch_time": 0.5
        }
        pipeline.orchestrator.get_metrics = AsyncMock(return_value=mock_orchestrator_metrics)
        
        # Simulate some pipeline operations
        pipeline._performance_metrics["total_requests"] = 10
        pipeline._performance_metrics["batch_requests"] = 3
        pipeline._performance_metrics["individual_requests"] = 7
        
        metrics = pipeline.get_performance_metrics()
        
        assert "pipeline_metrics" in metrics
        assert "orchestrator_metrics" in metrics
        assert metrics["pipeline_metrics"]["total_requests"] == 10
        assert metrics["orchestrator_metrics"]["cache_hit_rate"] == 0.75


@pytest.mark.integration
@pytest.mark.batching
@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance improvements through integration."""
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual_performance(self, batching_config, mock_redis_batching, sample_texts):
        """Test performance comparison between batched and individual processing."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock faster batch processing vs individual
        async def mock_batch_process(texts):
            await asyncio.sleep(0.01 * len(texts) / 4)  # Batch efficiency
            return [np.random.rand(128).astype(np.float32) for _ in texts]
        
        async def mock_individual_process(text):
            await asyncio.sleep(0.01)  # Individual processing time
            return np.random.rand(128).astype(np.float32)
        
        pipeline.orchestrator.process_embeddings_batch = mock_batch_process
        pipeline.orchestrator.start = AsyncMock()
        
        # Test batch processing time
        texts = sample_texts[:8]  # 8 texts
        
        start_time = time.time()
        batch_results = await pipeline.generate_embeddings(texts, use_batching=True)
        batch_time = time.time() - start_time
        
        # Simulate individual processing
        start_time = time.time()
        individual_results = []
        for text in texts:
            result = await mock_individual_process(text)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Batch should be faster
        assert len(batch_results) == len(individual_results) == len(texts)
        assert batch_time < individual_time, f"Batch time {batch_time:.3f}s should be < individual time {individual_time:.3f}s"
        
        # Calculate performance improvement
        improvement = (individual_time - batch_time) / individual_time
        assert improvement > 0.3, f"Performance improvement {improvement:.2%} should be > 30%"
    
    @pytest.mark.asyncio
    async def test_cache_performance_benefits(self, batching_config, mock_redis_batching):
        """Test cache performance benefits."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock cache behavior
        cached_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        # First call - cache miss (slower)
        pipeline.orchestrator.queue_embedding_request = AsyncMock(return_value="req1")
        pipeline.orchestrator.get_embedding_result = AsyncMock(return_value=cached_embedding)
        pipeline.orchestrator.start = AsyncMock()
        
        start_time = time.time()
        result1 = await pipeline.generate_single_embedding("test text")
        first_time = time.time() - start_time
        
        # Second call - simulate cache hit (faster)
        async def fast_cache_response(request_id, timeout=None):
            return cached_embedding  # Immediate return
        
        pipeline.orchestrator.get_embedding_result = fast_cache_response
        
        start_time = time.time()
        result2 = await pipeline.generate_single_embedding("test text")
        second_time = time.time() - start_time
        
        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
        
        # Second call should be faster (cache hit)
        assert second_time <= first_time, "Cache hit should be faster or equal"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, batching_config, mock_redis_batching, sample_texts):
        """Test handling of concurrent requests."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock concurrent processing
        async def mock_process_batch(texts):
            await asyncio.sleep(0.1)  # Simulate processing time
            return [np.random.rand(128).astype(np.float32) for _ in texts]
        
        pipeline.orchestrator.process_embeddings_batch = mock_process_batch
        pipeline.orchestrator.start = AsyncMock()
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(5):
            texts = sample_texts[:3]  # Small batches
            task = pipeline.generate_embeddings(texts)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all requests completed successfully
        assert len(results) == 5
        for result in results:
            assert len(result) == 3
            assert all(isinstance(emb, np.ndarray) for emb in result)
        
        # Should complete in reasonable time due to batching
        assert total_time < 1.0, f"Concurrent processing took {total_time:.3f}s, should be < 1.0s"


@pytest.mark.integration
@pytest.mark.batching
class TestRealWorldIntegration:
    """Test integration with real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_lightrag_patching(self, batching_config, mock_redis_batching, mock_lightrag_instance):
        """Test patching LightRAG instance with batching."""
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=True
        )
        
        with patch('lightrag.pipeline.batch_integration.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_batching
            
            await integration.initialize()
            
            # Patch the LightRAG instance
            await integration.patch_lightrag_embeddings(mock_lightrag_instance)
            
            # Verify the embedding function was patched
            assert hasattr(mock_lightrag_instance, 'embedding_func')
            
            # Test the patched function
            mock_texts = ["test text 1", "test text 2"]
            
            # Mock the enhanced pipeline
            if integration._enhanced_pipeline:
                integration._enhanced_pipeline.generate_embeddings = AsyncMock(
                    return_value=[np.random.rand(128).astype(np.float32) for _ in mock_texts]
                )
                
                # Call the patched function
                result = await mock_lightrag_instance.embedding_func(mock_texts)
                
                assert len(result) == len(mock_texts)
    
    @pytest.mark.asyncio
    async def test_enable_lightrag_batching_convenience_function(self, batching_config, mock_lightrag_instance):
        """Test the convenience function for enabling batching."""
        # Mock Redis availability check
        with patch('lightrag.pipeline.batch_integration.is_batching_available', return_value=False):
            # Should work with batching disabled
            integration = await enable_lightrag_batching(
                mock_lightrag_instance,
                config=batching_config
            )
            
            assert isinstance(integration, LightRAGBatchIntegration)
            assert integration.enable_batching is False
    
    @pytest.mark.asyncio
    async def test_configuration_optimization(self):
        """Test configuration optimization features."""
        # Test recommended configuration
        config = get_recommended_config()
        
        assert hasattr(config, 'batch_size')
        assert hasattr(config, 'ollama_model')
        assert config.batch_size > 0
        assert config.cache_ttl > 0
    
    def test_batching_availability_check(self):
        """Test batching availability checking."""
        # This should work without Redis connection
        availability = is_batching_available()
        
        # Should return boolean
        assert isinstance(availability, bool)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, batching_config, mock_redis_batching):
        """Test error handling in integration scenarios."""
        pipeline = EnhancedEmbeddingPipeline(
            redis_client=mock_redis_batching,
            config=batching_config
        )
        
        # Mock Redis failure
        mock_redis_batching.ping.side_effect = Exception("Redis connection failed")
        
        # Pipeline should handle Redis failure gracefully
        await pipeline.start()
        
        # Should still be able to process (with fallback)
        pipeline.orchestrator.process_embeddings_batch = AsyncMock(
            side_effect=RuntimeError("Processing failed")
        )
        
        # With fallback function
        def fallback_func(texts):
            return [np.array([1.0, 2.0], dtype=np.float32) for _ in texts]
        
        pipeline.fallback_func = fallback_func
        
        result = await pipeline.generate_embeddings(["test text"])
        
        assert len(result) == 1
        # Check that fallback was used (result should be from fallback function)
        assert isinstance(result[0], np.ndarray)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])


@pytest.mark.integration
@pytest.mark.batching
@pytest.mark.slow
class TestEndToEndIntegration:
    """End-to-end integration tests with full workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, batching_config, mock_redis_batching, sample_texts):
        """Test complete workflow from initialization to cleanup."""
        # Step 1: Initialize integration
        integration = LightRAGBatchIntegration(
            config=batching_config,
            enable_batching=True
        )
        
        with patch('lightrag.pipeline.batch_integration.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_batching
            
            try:
                # Step 2: Initialize
                await integration.initialize()
                assert integration._initialized is True
                
                # Step 3: Health check
                health = await integration.health_check()
                assert health["integration_status"] in ["healthy", "degraded"]
                
                # Step 4: Test embedding pipeline
                if integration._enhanced_pipeline:
                    # Mock pipeline operations
                    mock_embeddings = [np.random.rand(128).astype(np.float32) for _ in sample_texts]
                    integration._enhanced_pipeline.orchestrator.process_embeddings_batch = AsyncMock(
                        return_value=mock_embeddings
                    )
                    integration._enhanced_pipeline.orchestrator.start = AsyncMock()
                    
                    # Process embeddings
                    results = await integration._enhanced_pipeline.generate_embeddings(sample_texts)
                    assert len(results) == len(sample_texts)
                
                # Step 5: Get metrics
                metrics = integration.get_performance_metrics()
                assert "pipeline_metrics" in metrics or "error" in metrics
                
                # Step 6: Configuration summary
                summary = integration.get_configuration_summary()
                assert "batching_enabled" in summary
                
            finally:
                # Step 7: Cleanup
                await integration.cleanup()
                assert integration._initialized is False
    
    @pytest.mark.asyncio
    async def test_integration_with_async_context_manager(self, batching_config, mock_redis_batching):
        """Test integration using async context manager."""
        with patch('lightrag.pipeline.batch_integration.redis.Redis') as mock_redis_class:
            mock_redis_class.return_value = mock_redis_batching
            
            async with LightRAGBatchIntegration(config=batching_config) as integration:
                assert integration._initialized is True
                
                # Test functionality within context
                summary = integration.get_configuration_summary()
                assert "batching_enabled" in summary
                
                health = await integration.health_check()
                assert "integration_status" in health
            
            # Should be cleaned up after context exit
            assert integration._initialized is False


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-m", "integration and batching"])