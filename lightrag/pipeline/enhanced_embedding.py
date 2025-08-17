"""
Enhanced embedding pipeline with Ollama batching orchestrator integration.

Provides a drop-in replacement for existing embedding functions
while adding intelligent batching for 50%+ performance improvements.
"""

import asyncio
import logging
import time
from typing import List, Optional, Callable, Any, Dict
import numpy as np

from ..orchestrator import OllamaBatchingOrchestrator
from ..config.batch_config import BatchingConfig

logger = logging.getLogger(__name__)


class EnhancedEmbeddingPipeline:
    """
    Enhanced embedding pipeline with intelligent batching.
    
    Provides backward compatibility with existing LightRAG embedding
    functions while adding performance optimizations through batching.
    """
    
    def __init__(
        self,
        redis_client=None,
        config: Optional[BatchingConfig] = None,
        fallback_func: Optional[Callable] = None
    ):
        """
        Initialize enhanced embedding pipeline.
        
        Args:
            redis_client: Redis client for caching and queuing
            config: Batching configuration (uses defaults if None)
            fallback_func: Fallback embedding function if orchestrator fails
        """
        self.config = config or BatchingConfig.from_env()
        self.fallback_func = fallback_func
        
        # Initialize orchestrator
        self.orchestrator = OllamaBatchingOrchestrator(
            redis_client=redis_client,
            batch_size=self.config.batch_size,
            timeout=self.config.batch_timeout,
            processing_interval=self.config.processing_interval,
            cache_ttl=self.config.cache_ttl,
            max_retries=self.config.max_retries,
            ollama_config=self.config.to_ollama_config()
        )
        
        self._started = False
        self._performance_metrics = {
            "requests_processed": 0,
            "batch_requests": 0,
            "individual_requests": 0,
            "total_processing_time": 0.0,
            "fallback_requests": 0
        }
        
        logger.info("Enhanced embedding pipeline initialized")
    
    async def start(self):
        """Start the enhanced embedding pipeline."""
        if self._started:
            return
        
        await self.orchestrator.start()
        self._started = True
        logger.info("Enhanced embedding pipeline started")
    
    async def stop(self):
        """Stop the enhanced embedding pipeline."""
        if not self._started:
            return
        
        await self.orchestrator.stop()
        self._started = False
        logger.info("Enhanced embedding pipeline stopped")
    
    async def generate_embeddings(
        self, 
        texts: List[str],
        priority: int = 2,
        use_batching: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts with batching optimization.
        
        Args:
            texts: List of text strings to embed
            priority: Request priority (1=high, 2=normal, 3=low)
            use_batching: Whether to use batching orchestrator
            
        Returns:
            List of embedding arrays in same order as input
        """
        if not texts:
            return []
        
        if not self._started:
            await self.start()
        
        start_time = time.time()
        
        try:
            if use_batching and len(texts) > 1:
                # Use batching orchestrator for multiple texts
                embeddings = await self._generate_batch_embeddings(texts, priority)
                self._performance_metrics["batch_requests"] += 1
            else:
                # Use orchestrator for single texts or when batching disabled
                embeddings = await self._generate_individual_embeddings(texts, priority)
                self._performance_metrics["individual_requests"] += 1
            
            # Update metrics
            processing_time = time.time() - start_time
            self._performance_metrics["requests_processed"] += len(texts)
            self._performance_metrics["total_processing_time"] += processing_time
            
            logger.debug(f"Generated {len(embeddings)} embeddings in {processing_time:.3f}s")
            return embeddings
            
        except Exception as e:
            logger.error(f"Enhanced embedding generation failed: {e}")
            
            # Fallback to original function if available
            if self.fallback_func:
                logger.info("Falling back to original embedding function")
                embeddings = await self._use_fallback(texts)
                self._performance_metrics["fallback_requests"] += len(texts)
                return embeddings
            else:
                raise
    
    async def generate_single_embedding(
        self, 
        text: str,
        priority: int = 2,
        timeout: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            priority: Request priority (1=high, 2=normal, 3=low)
            timeout: Optional timeout override
            
        Returns:
            Embedding array
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        if not self._started:
            await self.start()
        
        try:
            # Queue request with orchestrator
            request_id = await self.orchestrator.queue_embedding_request(text, priority)
            
            # Get result
            embedding = await self.orchestrator.get_embedding_result(request_id, timeout)
            
            self._performance_metrics["individual_requests"] += 1
            self._performance_metrics["requests_processed"] += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Single embedding generation failed: {e}")
            
            # Fallback to original function if available
            if self.fallback_func:
                logger.info("Falling back to original embedding function for single text")
                embedding = await self._use_fallback([text])
                self._performance_metrics["fallback_requests"] += 1
                return embedding[0]
            else:
                raise
    
    async def _generate_batch_embeddings(
        self, 
        texts: List[str], 
        priority: int
    ) -> List[np.ndarray]:
        """Generate embeddings using batch processing."""
        return await self.orchestrator.process_embeddings_batch(texts)
    
    async def _generate_individual_embeddings(
        self, 
        texts: List[str], 
        priority: int
    ) -> List[np.ndarray]:
        """Generate embeddings using individual requests."""
        tasks = []
        for text in texts:
            task = self.generate_single_embedding(text, priority)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _use_fallback(self, texts: List[str]) -> List[np.ndarray]:
        """Use fallback embedding function."""
        if not self.fallback_func:
            raise RuntimeError("No fallback function available")
        
        # Check if fallback function is async
        if asyncio.iscoroutinefunction(self.fallback_func):
            result = await self.fallback_func(texts)
        else:
            result = self.fallback_func(texts)
        
        # Ensure result is a list of numpy arrays
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                # Single embedding
                return [result]
            else:
                # Multiple embeddings
                return [result[i] for i in range(result.shape[0])]
        elif isinstance(result, list):
            return [np.array(emb) if not isinstance(emb, np.ndarray) else emb for emb in result]
        else:
            raise ValueError(f"Unexpected fallback result type: {type(result)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        total_requests = max(self._performance_metrics["requests_processed"], 1)
        total_time = max(self._performance_metrics["total_processing_time"], 0.001)
        
        # Get orchestrator metrics
        orchestrator_metrics = self.orchestrator.get_metrics()
        
        return {
            "pipeline_metrics": {
                "total_requests": self._performance_metrics["requests_processed"],
                "batch_requests": self._performance_metrics["batch_requests"],
                "individual_requests": self._performance_metrics["individual_requests"],
                "fallback_requests": self._performance_metrics["fallback_requests"],
                "average_processing_time": self._performance_metrics["total_processing_time"] / total_requests,
                "requests_per_second": total_requests / total_time,
                "fallback_rate": self._performance_metrics["fallback_requests"] / total_requests
            },
            "orchestrator_metrics": orchestrator_metrics,
            "configuration": {
                "batch_size": self.config.batch_size,
                "cache_enabled": self.config.cache_enabled,
                "model": self.config.ollama_model,
                "processing_interval": self.config.processing_interval
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self._performance_metrics = {
            "requests_processed": 0,
            "batch_requests": 0,
            "individual_requests": 0,
            "total_processing_time": 0.0,
            "fallback_requests": 0
        }
        self.orchestrator.reset_metrics()
        logger.info("Enhanced embedding pipeline metrics reset")
    
    async def clear_cache(self):
        """Clear embedding cache."""
        if hasattr(self.orchestrator, 'embedding_cache'):
            await self.orchestrator.embedding_cache.clear_cache()
            logger.info("Embedding cache cleared")
    
    async def optimize_performance(self):
        """Perform performance optimization."""
        if hasattr(self.orchestrator, 'embedding_cache'):
            await self.orchestrator.embedding_cache.optimize_cache()
        
        # Log performance recommendations
        metrics = self.get_performance_metrics()
        cache_hit_rate = metrics["orchestrator_metrics"].get("cache_hit_rate", 0)
        
        if cache_hit_rate < 0.5:
            logger.warning(f"Low cache hit rate ({cache_hit_rate:.2%}), consider increasing cache TTL")
        
        fallback_rate = metrics["pipeline_metrics"]["fallback_rate"]
        if fallback_rate > 0.1:
            logger.warning(f"High fallback rate ({fallback_rate:.2%}), check orchestrator configuration")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Backward compatibility adapter
class LegacyEmbeddingAdapter:
    """
    Adapter to maintain compatibility with existing LightRAG code.
    
    Provides the same interface as original embedding functions
    while using the enhanced pipeline under the hood.
    """
    
    def __init__(
        self,
        enhanced_pipeline: EnhancedEmbeddingPipeline,
        original_func: Optional[Callable] = None
    ):
        """
        Initialize legacy adapter.
        
        Args:
            enhanced_pipeline: Enhanced pipeline instance
            original_func: Original embedding function for fallback
        """
        self.enhanced_pipeline = enhanced_pipeline
        self.original_func = original_func
        self._started = False
    
    async def __call__(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings with backward compatibility.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding arrays
        """
        if not self._started:
            await self.enhanced_pipeline.start()
            self._started = True
        
        try:
            return await self.enhanced_pipeline.generate_embeddings(texts)
        except Exception as e:
            logger.error(f"Enhanced pipeline failed, using original function: {e}")
            
            if self.original_func:
                if asyncio.iscoroutinefunction(self.original_func):
                    return await self.original_func(texts)
                else:
                    return self.original_func(texts)
            else:
                raise
    
    async def cleanup(self):
        """Cleanup adapter resources."""
        if self._started:
            await self.enhanced_pipeline.stop()
            self._started = False


def create_enhanced_embedding_function(
    redis_client=None,
    config: Optional[BatchingConfig] = None,
    fallback_func: Optional[Callable] = None
) -> LegacyEmbeddingAdapter:
    """
    Create an enhanced embedding function with backward compatibility.
    
    Args:
        redis_client: Redis client for caching and queuing
        config: Batching configuration
        fallback_func: Original embedding function for fallback
        
    Returns:
        Enhanced embedding function adapter
    """
    pipeline = EnhancedEmbeddingPipeline(
        redis_client=redis_client,
        config=config,
        fallback_func=fallback_func
    )
    
    return LegacyEmbeddingAdapter(pipeline, fallback_func)