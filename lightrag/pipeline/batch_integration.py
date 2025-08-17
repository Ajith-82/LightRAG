"""
Integration module for LightRAG with Ollama batching orchestrator.

Provides seamless integration with existing LightRAG pipeline
while enabling 50%+ performance improvements through batching.
"""

import asyncio
import logging
import os
from typing import Optional, Callable, Any, Dict
import redis.asyncio as redis

from ..orchestrator import OllamaBatchingOrchestrator
from ..config.batch_config import BatchingConfig
from .enhanced_embedding import EnhancedEmbeddingPipeline, create_enhanced_embedding_function

logger = logging.getLogger(__name__)


class LightRAGBatchIntegration:
    """
    Main integration class for LightRAG batching orchestrator.
    
    Handles initialization, configuration, and integration with
    existing LightRAG embedding pipeline.
    """
    
    def __init__(
        self,
        config: Optional[BatchingConfig] = None,
        redis_url: Optional[str] = None,
        enable_batching: bool = True
    ):
        """
        Initialize LightRAG batch integration.
        
        Args:
            config: Batching configuration (auto-detected if None)
            redis_url: Redis connection URL (auto-detected if None)
            enable_batching: Whether to enable batching (can be disabled for testing)
        """
        self.config = config or BatchingConfig.from_env()
        self.enable_batching = enable_batching
        self._redis_client = None
        self._enhanced_pipeline = None
        self._original_embedding_func = None
        self._initialized = False
        
        # Auto-detect Redis URL
        if redis_url is None:
            redis_url = self._build_redis_url()
        self.redis_url = redis_url
        
        logger.info(f"LightRAG batch integration initialized (batching {'enabled' if enable_batching else 'disabled'})")
    
    async def initialize(self):
        """Initialize Redis connection and enhanced pipeline."""
        if self._initialized:
            return
        
        if self.enable_batching:
            try:
                # Initialize Redis client
                self._redis_client = await self._create_redis_client()
                logger.info("Redis client initialized for batching")
                
                # Initialize enhanced pipeline
                self._enhanced_pipeline = EnhancedEmbeddingPipeline(
                    redis_client=self._redis_client,
                    config=self.config,
                    fallback_func=self._original_embedding_func
                )
                
                await self._enhanced_pipeline.start()
                logger.info("Enhanced embedding pipeline initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize batching components: {e}")
                logger.info("Continuing without batching optimization")
                self.enable_batching = False
        
        self._initialized = True
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._enhanced_pipeline:
            await self._enhanced_pipeline.stop()
            self._enhanced_pipeline = None
        
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
        
        self._initialized = False
        logger.info("LightRAG batch integration cleanup completed")
    
    def wrap_embedding_function(self, original_func: Callable) -> Callable:
        """
        Wrap an existing embedding function with batching optimization.
        
        Args:
            original_func: Original embedding function to wrap
            
        Returns:
            Enhanced embedding function with batching
        """
        self._original_embedding_func = original_func
        
        if not self.enable_batching:
            logger.info("Batching disabled, returning original function")
            return original_func
        
        # Create enhanced function
        enhanced_func = create_enhanced_embedding_function(
            redis_client=self._redis_client,
            config=self.config,
            fallback_func=original_func
        )
        
        logger.info("Embedding function wrapped with batching optimization")
        return enhanced_func
    
    async def patch_lightrag_embeddings(self, lightrag_instance):
        """
        Patch LightRAG instance to use batching orchestrator.
        
        Args:
            lightrag_instance: LightRAG instance to patch
        """
        if not self.enable_batching:
            logger.info("Batching disabled, skipping LightRAG patching")
            return
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if LightRAG has embedding_func
            if hasattr(lightrag_instance, 'embedding_func'):
                original_func = lightrag_instance.embedding_func
                
                # Create wrapper for async function
                async def enhanced_embedding_wrapper(texts):
                    if not self._enhanced_pipeline:
                        logger.warning("Enhanced pipeline not available, using original function")
                        return await original_func(texts)
                    
                    return await self._enhanced_pipeline.generate_embeddings(texts)
                
                # Replace the embedding function
                lightrag_instance.embedding_func = enhanced_embedding_wrapper
                logger.info("LightRAG embedding function patched with batching optimization")
            
            else:
                logger.warning("LightRAG instance does not have embedding_func attribute")
        
        except Exception as e:
            logger.error(f"Failed to patch LightRAG embeddings: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self._enhanced_pipeline:
            return {"error": "Enhanced pipeline not initialized"}
        
        return self._enhanced_pipeline.get_performance_metrics()
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "batching_enabled": self.enable_batching,
            "redis_url": self.redis_url,
            "config": {
                "batch_size": self.config.batch_size,
                "processing_interval": self.config.processing_interval,
                "cache_ttl": self.config.cache_ttl,
                "model": self.config.ollama_model,
                "max_retries": self.config.max_retries
            },
            "status": "initialized" if self._initialized else "not_initialized"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "integration_status": "healthy",
            "redis_status": "not_configured",
            "pipeline_status": "not_configured",
            "orchestrator_status": "not_configured"
        }
        
        if not self.enable_batching:
            health["integration_status"] = "disabled"
            return health
        
        # Check Redis connection
        if self._redis_client:
            try:
                await self._redis_client.ping()
                health["redis_status"] = "healthy"
            except Exception as e:
                health["redis_status"] = f"error: {e}"
                health["integration_status"] = "degraded"
        
        # Check enhanced pipeline
        if self._enhanced_pipeline:
            try:
                metrics = self._enhanced_pipeline.get_performance_metrics()
                health["pipeline_status"] = "healthy"
                health["orchestrator_status"] = "healthy"
                # Store metrics summary to avoid type issues
                health["performance_summary"] = {
                    "total_requests": metrics.get("pipeline_metrics", {}).get("total_requests", 0),
                    "cache_hit_rate": metrics.get("orchestrator_metrics", {}).get("cache_hit_rate", 0),
                    "batch_requests": metrics.get("pipeline_metrics", {}).get("batch_requests", 0)
                }
            except Exception as e:
                health["pipeline_status"] = f"error: {e}"
                health["integration_status"] = "degraded"
        
        return health
    
    async def _create_redis_client(self):
        """Create Redis client with connection pooling."""
        redis_config = self.config.to_redis_config()
        
        try:
            # Create connection pool
            pool = redis.ConnectionPool.from_url(
                self.redis_url,
                **redis_config
            )
            
            # Create Redis client
            client = redis.Redis(connection_pool=pool)
            
            # Test connection
            await client.ping()
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            raise
    
    def _build_redis_url(self) -> str:
        """Build Redis URL from configuration."""
        host = self.config.redis_host
        port = self.config.redis_port
        db = self.config.redis_db
        password = self.config.redis_password
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        else:
            return f"redis://{host}:{port}/{db}"
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Convenience functions for easy integration

async def enable_lightrag_batching(
    lightrag_instance,
    config: Optional[BatchingConfig] = None,
    redis_url: Optional[str] = None
) -> LightRAGBatchIntegration:
    """
    Enable batching optimization for a LightRAG instance.
    
    Args:
        lightrag_instance: LightRAG instance to optimize
        config: Batching configuration (auto-detected if None)
        redis_url: Redis connection URL (auto-detected if None)
        
    Returns:
        Integration instance for monitoring and control
    """
    integration = LightRAGBatchIntegration(config=config, redis_url=redis_url)
    await integration.initialize()
    await integration.patch_lightrag_embeddings(lightrag_instance)
    
    logger.info("LightRAG batching optimization enabled")
    return integration


def is_batching_available() -> bool:
    """
    Check if batching optimization is available.
    
    Returns:
        True if Redis is accessible and configuration is valid
    """
    try:
        config = BatchingConfig.from_env()
        
        # Check if Redis is available (simple connection test)
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((config.redis_host, config.redis_port))
        sock.close()
        
        return result == 0
        
    except Exception as e:
        logger.debug(f"Batching availability check failed: {e}")
        return False


def get_recommended_config() -> BatchingConfig:
    """
    Get recommended configuration based on detected environment.
    
    Returns:
        Optimized BatchingConfig for the current environment
    """
    config = BatchingConfig.from_env()
    
    # Auto-detect model and optimize
    model = config.ollama_model
    if model:
        config = config.get_optimized_config_for_model(model)
    
    # Environment-specific optimizations
    memory_gb = _get_available_memory_gb()
    if memory_gb >= 32:
        # High-memory environment (like HP DL380 Gen9)
        config.batch_size = min(32, config.max_batch_size)
        config.cache_max_memory_mb = 1000
        config.max_queue_size = 2000
        logger.info(f"Applied high-memory optimizations for {memory_gb}GB system")
    elif memory_gb >= 16:
        # Medium-memory environment
        config.batch_size = min(16, config.max_batch_size)
        config.cache_max_memory_mb = 500
        config.max_queue_size = 1000
        logger.info(f"Applied medium-memory optimizations for {memory_gb}GB system")
    else:
        # Low-memory environment
        config.batch_size = min(8, config.max_batch_size)
        config.cache_max_memory_mb = 200
        config.max_queue_size = 500
        logger.info(f"Applied low-memory optimizations for {memory_gb}GB system")
    
    return config


def _get_available_memory_gb() -> int:
    """Get available system memory in GB."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return int(memory.total / (1024**3))
    except ImportError:
        # Fallback: try to read from /proc/meminfo
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        memory_kb = int(line.split()[1])
                        return int(memory_kb / (1024**2))
        except Exception:
            pass
    
    # Default assumption
    return 8