"""
Redis-based request queue for Ollama batching orchestrator.

Manages queuing, prioritization, and retrieval of embedding requests
using Redis as the backend for persistence and coordination.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RequestQueue:
    """
    Redis-based queue for managing embedding requests.
    
    Features:
    - Priority-based queuing (high, normal, low)
    - Atomic batch dequeuing
    - Request persistence and recovery
    - Queue size monitoring
    """
    
    QUEUE_KEYS = {
        'high_priority': 'lightrag:embedding:requests:high',
        'normal_priority': 'lightrag:embedding:requests:normal', 
        'low_priority': 'lightrag:embedding:requests:low',
        'processing': 'lightrag:embedding:processing',
        'failed': 'lightrag:embedding:failed'
    }
    
    def __init__(self, redis_client=None):
        """
        Initialize request queue.
        
        Args:
            redis_client: Redis client instance (None for in-memory fallback)
        """
        self.redis = redis_client
        self._in_memory_queues = {
            'high_priority': [],
            'normal_priority': [],
            'low_priority': [],
            'processing': [],
            'failed': []
        }
        self._initialized = False
        
        if self.redis is None:
            logger.warning("No Redis client provided, using in-memory queue (not persistent)")
    
    async def initialize(self):
        """Initialize the queue system."""
        if self._initialized:
            return
            
        if self.redis:
            try:
                # Test Redis connection
                await self.redis.ping()
                logger.info("Redis queue initialized successfully")
            except Exception as e:
                logger.error(f"Redis connection failed, falling back to in-memory: {e}")
                self.redis = None
        
        self._initialized = True
    
    async def enqueue_request(self, request_data: Dict[str, Any]):
        """
        Add a request to the appropriate priority queue.
        
        Args:
            request_data: Dictionary containing request information
        """
        if not self._initialized:
            await self.initialize()
        
        priority = request_data.get("priority", 2)
        queue_key = self._get_queue_key_by_priority(priority)
        
        if self.redis:
            try:
                # Add to Redis list
                serialized_request = json.dumps(request_data)
                await self.redis.lpush(queue_key, serialized_request)
                logger.debug(f"Queued request {request_data['request_id']} to {queue_key}")
            except Exception as e:
                logger.error(f"Failed to queue request to Redis: {e}")
                # Fallback to in-memory
                self._enqueue_memory(request_data, priority)
        else:
            self._enqueue_memory(request_data, priority)
    
    async def dequeue_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Dequeue a batch of requests for processing.
        
        Args:
            batch_size: Maximum number of requests to dequeue
            
        Returns:
            List of request dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        requests = []
        
        if self.redis:
            try:
                # Dequeue from Redis with priority order
                for priority in [1, 2, 3]:  # High to low priority
                    if len(requests) >= batch_size:
                        break
                        
                    queue_key = self._get_queue_key_by_priority(priority)
                    remaining_capacity = batch_size - len(requests)
                    
                    # Use pipeline for atomic operations
                    pipe = self.redis.pipeline()
                    pipe.lrange(queue_key, 0, remaining_capacity - 1)
                    pipe.ltrim(queue_key, remaining_capacity, -1)
                    results = await pipe.execute()
                    
                    if results[0]:  # If we got requests
                        for serialized_request in results[0]:
                            try:
                                request = json.loads(serialized_request)
                                requests.append(request)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to deserialize request: {e}")
                
            except Exception as e:
                logger.error(f"Failed to dequeue from Redis: {e}")
                # Fallback to in-memory
                requests = self._dequeue_memory_batch(batch_size)
        else:
            requests = self._dequeue_memory_batch(batch_size)
        
        if requests:
            logger.debug(f"Dequeued batch of {len(requests)} requests")
        
        return requests
    
    async def requeue_failed_requests(self, requests: List[Dict[str, Any]]):
        """
        Requeue failed requests for retry.
        
        Args:
            requests: List of failed request dictionaries
        """
        for request in requests:
            request["retry_count"] = request.get("retry_count", 0) + 1
            request["retry_timestamp"] = time.time()
            
            # Requeue with lower priority for retries
            original_priority = request.get("priority", 2)
            retry_priority = min(original_priority + 1, 3)  # Demote priority
            request["priority"] = retry_priority
            
            await self.enqueue_request(request)
    
    async def move_to_failed(self, requests: List[Dict[str, Any]]):
        """
        Move requests to failed queue for analysis.
        
        Args:
            requests: List of permanently failed requests
        """
        if self.redis:
            try:
                for request in requests:
                    request["failed_timestamp"] = time.time()
                    serialized_request = json.dumps(request)
                    await self.redis.lpush(self.QUEUE_KEYS['failed'], serialized_request)
            except Exception as e:
                logger.error(f"Failed to move requests to failed queue: {e}")
        else:
            for request in requests:
                request["failed_timestamp"] = time.time()
                self._in_memory_queues['failed'].append(request)
    
    async def get_queue_sizes(self) -> Dict[str, int]:
        """
        Get the current size of all queues.
        
        Returns:
            Dictionary mapping queue names to sizes
        """
        sizes = {}
        
        if self.redis:
            try:
                for queue_name, queue_key in self.QUEUE_KEYS.items():
                    size = await self.redis.llen(queue_key)
                    sizes[queue_name] = size
            except Exception as e:
                logger.error(f"Failed to get queue sizes from Redis: {e}")
                sizes = {name: len(queue) for name, queue in self._in_memory_queues.items()}
        else:
            sizes = {name: len(queue) for name, queue in self._in_memory_queues.items()}
        
        return sizes
    
    async def size(self) -> int:
        """Get total number of pending requests."""
        sizes = await self.get_queue_sizes()
        return sizes.get('high_priority', 0) + sizes.get('normal_priority', 0) + sizes.get('low_priority', 0)
    
    async def clear_queues(self, queue_names: Optional[List[str]] = None):
        """
        Clear specified queues or all queues.
        
        Args:
            queue_names: List of queue names to clear (None for all)
        """
        if queue_names is None:
            queue_names = list(self.QUEUE_KEYS.keys())
        
        if self.redis:
            try:
                for queue_name in queue_names:
                    if queue_name in self.QUEUE_KEYS:
                        await self.redis.delete(self.QUEUE_KEYS[queue_name])
                logger.info(f"Cleared Redis queues: {queue_names}")
            except Exception as e:
                logger.error(f"Failed to clear Redis queues: {e}")
        
        # Also clear in-memory queues
        for queue_name in queue_names:
            if queue_name in self._in_memory_queues:
                self._in_memory_queues[queue_name].clear()
    
    async def cleanup(self):
        """Cleanup queue resources."""
        # Note: We don't close Redis connection as it might be shared
        self._initialized = False
        logger.info("Request queue cleanup completed")
    
    def _get_queue_key_by_priority(self, priority: int) -> str:
        """Get Redis queue key by priority level."""
        if priority == 1:
            return self.QUEUE_KEYS['high_priority']
        elif priority == 3:
            return self.QUEUE_KEYS['low_priority']
        else:
            return self.QUEUE_KEYS['normal_priority']
    
    def _enqueue_memory(self, request_data: Dict[str, Any], priority: int):
        """Enqueue request to in-memory queue."""
        if priority == 1:
            self._in_memory_queues['high_priority'].append(request_data)
        elif priority == 3:
            self._in_memory_queues['low_priority'].append(request_data)
        else:
            self._in_memory_queues['normal_priority'].append(request_data)
    
    def _dequeue_memory_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Dequeue batch from in-memory queues."""
        requests = []
        
        # Process high priority first
        while len(requests) < batch_size and self._in_memory_queues['high_priority']:
            requests.append(self._in_memory_queues['high_priority'].pop(0))
        
        # Then normal priority
        while len(requests) < batch_size and self._in_memory_queues['normal_priority']:
            requests.append(self._in_memory_queues['normal_priority'].pop(0))
        
        # Finally low priority
        while len(requests) < batch_size and self._in_memory_queues['low_priority']:
            requests.append(self._in_memory_queues['low_priority'].pop(0))
        
        return requests