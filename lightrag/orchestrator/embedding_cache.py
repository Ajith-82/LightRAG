"""
Redis-based embedding cache for Ollama batching orchestrator.

Provides intelligent caching of computed embeddings to achieve
75% cache hit rate and reduce duplicate computations.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Redis-based cache for storing and retrieving computed embeddings.
    
    Features:
    - TTL-based expiration
    - Efficient text hashing
    - Compressed storage using pickle
    - Cache hit/miss statistics
    - Memory fallback when Redis unavailable
    """
    
    def __init__(self, redis_client=None, db: int = 2, ttl: int = 3600):
        """
        Initialize embedding cache.
        
        Args:
            redis_client: Redis client instance (None for in-memory fallback)
            db: Redis database number (default: 2)
            ttl: Time-to-live for cached embeddings in seconds (default: 3600)
        """
        self.redis = redis_client
        self.db = db
        self.ttl = ttl
        self._in_memory_cache = {}  # Fallback cache
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0
        }
        self._initialized = False
        
        if self.redis is None:
            logger.warning("No Redis client provided, using in-memory cache (not persistent)")
    
    async def initialize(self):
        """Initialize the cache system."""
        if self._initialized:
            return
            
        if self.redis:
            try:
                # Test Redis connection and select database
                await self.redis.ping()
                await self.redis.select(self.db)
                logger.info(f"Redis embedding cache initialized on DB {self.db}")
            except Exception as e:
                logger.error(f"Redis cache initialization failed, falling back to in-memory: {e}")
                self.redis = None
        
        self._initialized = True
    
    async def get_text_hash(self, text: str) -> str:
        """
        Generate a consistent hash for text content.
        
        Args:
            text: Input text to hash
            
        Returns:
            Hash string for the text
        """
        # Normalize text for consistent hashing
        normalized_text = text.strip().lower()
        
        # Use SHA-256 for consistent, collision-resistant hashing
        hash_object = hashlib.sha256(normalized_text.encode('utf-8'))
        return hash_object.hexdigest()
    
    async def get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Retrieve a cached embedding by text hash.
        
        Args:
            text_hash: Hash of the text to look up
            
        Returns:
            Cached embedding array or None if not found
        """
        if not self._initialized:
            await self.initialize()
        
        cache_key = f"embedding:{text_hash}"
        
        if self.redis:
            try:
                # Get from Redis
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    # Deserialize the embedding
                    embedding = pickle.loads(cached_data)
                    self._cache_stats["hits"] += 1
                    logger.debug(f"Cache hit for hash {text_hash[:8]}...")
                    return embedding
                else:
                    self._cache_stats["misses"] += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Redis cache retrieval failed: {e}")
                # Fallback to in-memory cache
                return self._get_from_memory_cache(text_hash)
        else:
            return self._get_from_memory_cache(text_hash)
    
    async def cache_embedding(self, text_hash: str, embedding: np.ndarray):
        """
        Cache an embedding with TTL.
        
        Args:
            text_hash: Hash of the text
            embedding: Computed embedding array to cache
        """
        if not self._initialized:
            await self.initialize()
        
        cache_key = f"embedding:{text_hash}"
        
        if self.redis:
            try:
                # Serialize and store in Redis with TTL
                serialized_embedding = pickle.dumps(embedding)
                await self.redis.setex(cache_key, self.ttl, serialized_embedding)
                self._cache_stats["stores"] += 1
                logger.debug(f"Cached embedding for hash {text_hash[:8]}... (TTL: {self.ttl}s)")
                
            except Exception as e:
                logger.error(f"Redis cache storage failed: {e}")
                # Fallback to in-memory cache
                self._store_in_memory_cache(text_hash, embedding)
        else:
            self._store_in_memory_cache(text_hash, embedding)
    
    async def invalidate_cache(self, text_hash: str):
        """
        Remove a specific embedding from cache.
        
        Args:
            text_hash: Hash of the text to remove
        """
        cache_key = f"embedding:{text_hash}"
        
        if self.redis:
            try:
                result = await self.redis.delete(cache_key)
                if result:
                    logger.debug(f"Invalidated cache for hash {text_hash[:8]}...")
            except Exception as e:
                logger.error(f"Redis cache invalidation failed: {e}")
        
        # Also remove from in-memory cache
        self._in_memory_cache.pop(text_hash, None)
    
    async def clear_cache(self):
        """Clear all cached embeddings."""
        if self.redis:
            try:
                # Use pattern matching to delete all embedding keys
                pattern = "embedding:*"
                cursor = 0
                deleted_count = 0
                
                while True:
                    cursor, keys = await self.redis.scan(cursor=cursor, match=pattern)
                    if keys:
                        deleted = await self.redis.delete(*keys)
                        deleted_count += deleted
                    
                    if cursor == 0:
                        break
                
                logger.info(f"Cleared {deleted_count} embeddings from Redis cache")
                
            except Exception as e:
                logger.error(f"Redis cache clearing failed: {e}")
        
        # Clear in-memory cache
        self._in_memory_cache.clear()
        logger.info("Cleared in-memory embedding cache")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / max(total_requests, 1)
        
        stats = {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "stores": self._cache_stats["stores"],
            "evictions": self._cache_stats["evictions"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "miss_rate": 1.0 - hit_rate,
            "cache_backend": "redis" if self.redis else "memory"
        }
        
        # Add Redis-specific stats if available
        if self.redis:
            try:
                # Get cache size from Redis
                pattern = "embedding:*"
                cursor = 0
                cache_size = 0
                
                while True:
                    cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=1000)
                    cache_size += len(keys)
                    if cursor == 0:
                        break
                
                stats["cache_size"] = cache_size
                stats["memory_usage_mb"] = cache_size * 0.004  # Rough estimate
                
            except Exception as e:
                logger.error(f"Failed to get Redis cache stats: {e}")
                stats["cache_size"] = "unknown"
        else:
            stats["cache_size"] = len(self._in_memory_cache)
            stats["memory_usage_mb"] = len(self._in_memory_cache) * 0.004
        
        return stats
    
    async def optimize_cache(self):
        """
        Perform cache optimization and cleanup.
        
        This method can be called periodically to:
        - Remove expired entries (handled automatically by Redis TTL)
        - Compact memory usage
        - Update statistics
        """
        if self.redis:
            try:
                # Redis handles TTL automatically, but we can check for stale data
                # This is mainly for monitoring and statistics
                pattern = "embedding:*"
                cursor = 0
                total_keys = 0
                
                while True:
                    cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=1000)
                    total_keys += len(keys)
                    if cursor == 0:
                        break
                
                logger.info(f"Cache optimization completed. Total embeddings: {total_keys}")
                
            except Exception as e:
                logger.error(f"Cache optimization failed: {e}")
        else:
            # For in-memory cache, implement simple LRU eviction
            max_size = 10000  # Maximum number of cached embeddings
            if len(self._in_memory_cache) > max_size:
                # Remove oldest entries (simple approach)
                items_to_remove = len(self._in_memory_cache) - max_size
                keys_to_remove = list(self._in_memory_cache.keys())[:items_to_remove]
                
                for key in keys_to_remove:
                    del self._in_memory_cache[key]
                    self._cache_stats["evictions"] += 1
                
                logger.info(f"Evicted {items_to_remove} entries from in-memory cache")
    
    def reset_stats(self):
        """Reset cache statistics."""
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0
        }
        logger.info("Cache statistics reset")
    
    async def cleanup(self):
        """Cleanup cache resources."""
        # Note: We don't close Redis connection as it might be shared
        self._initialized = False
        logger.info("Embedding cache cleanup completed")
    
    def _get_from_memory_cache(self, text_hash: str) -> Optional[np.ndarray]:
        """Get embedding from in-memory cache."""
        if text_hash in self._in_memory_cache:
            entry = self._in_memory_cache[text_hash]
            
            # Check TTL for in-memory cache
            if time.time() - entry["timestamp"] < self.ttl:
                self._cache_stats["hits"] += 1
                logger.debug(f"Memory cache hit for hash {text_hash[:8]}...")
                return entry["embedding"]
            else:
                # Expired entry
                del self._in_memory_cache[text_hash]
                self._cache_stats["evictions"] += 1
        
        self._cache_stats["misses"] += 1
        return None
    
    def _store_in_memory_cache(self, text_hash: str, embedding: np.ndarray):
        """Store embedding in in-memory cache."""
        self._in_memory_cache[text_hash] = {
            "embedding": embedding,
            "timestamp": time.time()
        }
        self._cache_stats["stores"] += 1
        logger.debug(f"Stored embedding in memory cache for hash {text_hash[:8]}...")