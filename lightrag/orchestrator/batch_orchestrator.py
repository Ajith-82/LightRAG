"""
Main Ollama Batching Orchestrator for LightRAG.

Coordinates embedding request batching, processing, and result delivery
to achieve significant performance improvements over individual requests.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .request_queue import RequestQueue
from .embedding_cache import EmbeddingCache
from .batch_processor import BatchProcessor
from .error_handler import BatchProcessingErrorHandler

logger = logging.getLogger(__name__)


class OllamaBatchingOrchestrator:
    """
    Main orchestrator for batching Ollama embedding requests.
    
    Provides intelligent batching to improve performance by 50%+ through:
    - Request queuing and batching
    - Intelligent caching with Redis
    - Optimized Ollama API utilization
    - Comprehensive error handling and retry logic
    """
    
    def __init__(
        self,
        redis_client=None,
        batch_size: int = 32,
        timeout: int = 30000,
        processing_interval: float = 0.1,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        ollama_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the batching orchestrator.
        
        Args:
            redis_client: Redis client instance for queuing and caching
            batch_size: Maximum number of requests per batch (default: 32)
            timeout: Request timeout in milliseconds (default: 30000)
            processing_interval: Batch processing interval in seconds (default: 0.1)
            cache_ttl: Cache time-to-live in seconds (default: 3600)
            max_retries: Maximum retry attempts for failed requests (default: 3)
            ollama_config: Ollama-specific configuration
        """
        self.batch_size = batch_size
        self.timeout = timeout / 1000.0  # Convert to seconds
        self.processing_interval = processing_interval
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.ollama_config = ollama_config or {}
        
        # Initialize components
        self.request_queue = RequestQueue(redis_client)
        self.embedding_cache = EmbeddingCache(redis_client, db=2, ttl=cache_ttl)
        self.batch_processor = BatchProcessor(ollama_config)
        self.error_handler = BatchProcessingErrorHandler(max_retries)
        
        # Runtime state
        self._processing_task = None
        self._active_requests: Dict[str, asyncio.Future] = {}
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batches_processed": 0,
            "total_processing_time": 0.0,
            "failed_requests": 0
        }
        self._running = False
        
        logger.info(f"Initialized Ollama Batching Orchestrator with batch_size={batch_size}")
    
    async def start(self):
        """Start the orchestrator and background processing."""
        if self._running:
            logger.warning("Orchestrator already running")
            return
            
        self._running = True
        logger.info("Starting Ollama Batching Orchestrator")
        
        # Start background batch processing
        self._processing_task = asyncio.create_task(self._batch_processing_loop())
        
        # Initialize components
        await self.request_queue.initialize()
        await self.embedding_cache.initialize()
        
        logger.info("Ollama Batching Orchestrator started successfully")
    
    async def stop(self):
        """Stop the orchestrator and cleanup resources."""
        if not self._running:
            return
            
        logger.info("Stopping Ollama Batching Orchestrator")
        self._running = False
        
        # Cancel background processing
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Complete any pending requests
        for future in self._active_requests.values():
            if not future.done():
                future.cancel()
        
        # Cleanup components
        await self.request_queue.cleanup()
        await self.embedding_cache.cleanup()
        
        logger.info("Ollama Batching Orchestrator stopped")
    
    async def queue_embedding_request(
        self, 
        text: str, 
        priority: int = 2,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Queue an embedding request for batch processing.
        
        Args:
            text: Text to generate embedding for
            priority: Request priority (1=high, 2=normal, 3=low)
            metadata: Optional metadata for the request
            
        Returns:
            request_id: Unique identifier for tracking the request
        """
        request_id = str(uuid.uuid4())
        
        # Check cache first
        text_hash = await self.embedding_cache.get_text_hash(text)
        cached_embedding = await self.embedding_cache.get_cached_embedding(text_hash)
        
        if cached_embedding is not None:
            # Cache hit - return immediately
            self._metrics["cache_hits"] += 1
            self._metrics["total_requests"] += 1
            
            # Create a completed future
            future = asyncio.Future()
            future.set_result(cached_embedding)
            self._active_requests[request_id] = future
            
            logger.debug(f"Cache hit for request {request_id}")
            return request_id
        
        # Cache miss - queue for batch processing
        self._metrics["cache_misses"] += 1
        self._metrics["total_requests"] += 1
        
        request_data = {
            "request_id": request_id,
            "text": text,
            "text_hash": text_hash,
            "timestamp": time.time(),
            "priority": priority,
            "retry_count": 0,
            "metadata": metadata or {}
        }
        
        # Queue the request
        await self.request_queue.enqueue_request(request_data)
        
        # Create future for result tracking
        future = asyncio.Future()
        self._active_requests[request_id] = future
        
        logger.debug(f"Queued embedding request {request_id} (cache miss)")
        return request_id
    
    async def get_embedding_result(
        self, 
        request_id: str, 
        timeout: Optional[float] = None
    ) -> np.ndarray:
        """
        Get the result of an embedding request.
        
        Args:
            request_id: Request identifier from queue_embedding_request
            timeout: Optional timeout in seconds (uses orchestrator timeout if None)
            
        Returns:
            embedding: The computed embedding as numpy array
            
        Raises:
            TimeoutError: If request times out
            RuntimeError: If request processing failed
        """
        if request_id not in self._active_requests:
            raise ValueError(f"Unknown request ID: {request_id}")
        
        future = self._active_requests[request_id]
        request_timeout = timeout or self.timeout
        
        try:
            result = await asyncio.wait_for(future, timeout=request_timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out after {request_timeout}s")
            raise TimeoutError(f"Embedding request timed out: {request_id}")
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise RuntimeError(f"Embedding request failed: {e}")
        finally:
            # Cleanup completed request
            self._active_requests.pop(request_id, None)
    
    async def process_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Process multiple texts with automatic batching optimization.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            embeddings: List of embedding arrays in same order as input
        """
        if not texts:
            return []
        
        # Queue all requests
        request_ids = []
        for text in texts:
            request_id = await self.queue_embedding_request(text)
            request_ids.append(request_id)
        
        # Collect results
        embeddings = []
        for request_id in request_ids:
            try:
                embedding = await self.get_embedding_result(request_id)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to get embedding for request {request_id}: {e}")
                # Return zero embedding for failed requests
                embedding_dim = self.ollama_config.get("embedding_dim", 1024)
                embeddings.append(np.zeros(embedding_dim))
        
        return embeddings
    
    async def _batch_processing_loop(self):
        """Background loop for processing queued requests in batches."""
        logger.info("Started batch processing loop")
        
        while self._running:
            try:
                await self._process_next_batch()
                await asyncio.sleep(self.processing_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
        
        logger.info("Batch processing loop stopped")
    
    async def _process_next_batch(self):
        """Process the next batch of queued requests."""
        # Get batch of requests from queue
        requests = await self.request_queue.dequeue_batch(self.batch_size)
        
        if not requests:
            return  # No requests to process
        
        logger.debug(f"Processing batch of {len(requests)} requests")
        batch_start_time = time.time()
        
        try:
            # Extract texts for batch processing
            texts = [req["text"] for req in requests]
            text_hashes = [req["text_hash"] for req in requests]
            request_ids = [req["request_id"] for req in requests]
            
            # Process batch with Ollama
            embeddings = await self.batch_processor.process_batch(texts)
            
            # Cache and deliver results
            for i, (request_id, text_hash, embedding) in enumerate(zip(request_ids, text_hashes, embeddings)):
                # Cache the embedding
                await self.embedding_cache.cache_embedding(text_hash, embedding)
                
                # Deliver result to waiting future
                if request_id in self._active_requests:
                    future = self._active_requests[request_id]
                    if not future.done():
                        future.set_result(embedding)
            
            # Update metrics
            batch_time = time.time() - batch_start_time
            self._metrics["batches_processed"] += 1
            self._metrics["total_processing_time"] += batch_time
            
            logger.debug(f"Processed batch of {len(requests)} requests in {batch_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self._metrics["failed_requests"] += len(requests)
            
            # Handle failed requests
            await self.error_handler.handle_batch_failure(requests, str(e))
            
            # Notify waiting futures of failure
            for request in requests:
                request_id = request["request_id"]
                if request_id in self._active_requests:
                    future = self._active_requests[request_id]
                    if not future.done():
                        future.set_exception(RuntimeError(f"Batch processing failed: {e}"))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        total_requests = max(self._metrics["total_requests"], 1)
        total_batches = max(self._metrics["batches_processed"], 1)
        
        return {
            "total_requests": self._metrics["total_requests"],
            "cache_hit_rate": self._metrics["cache_hits"] / total_requests,
            "cache_miss_rate": self._metrics["cache_misses"] / total_requests,
            "batches_processed": self._metrics["batches_processed"],
            "average_batch_time": self._metrics["total_processing_time"] / total_batches,
            "failed_requests": self._metrics["failed_requests"],
            "failure_rate": self._metrics["failed_requests"] / total_requests,
            "active_requests": len(self._active_requests),
            "queue_size": self.request_queue.size() if hasattr(self.request_queue, 'size') else 0
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batches_processed": 0,
            "total_processing_time": 0.0,
            "failed_requests": 0
        }
        logger.info("Performance metrics reset")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()