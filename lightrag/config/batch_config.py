"""
Configuration management for Ollama batching orchestrator.

Provides centralized configuration handling with environment variable
support and validation for optimal batch processing performance.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatchingConfig:
    """
    Configuration class for Ollama batching orchestrator.
    
    Provides optimal default values based on production testing
    with HP DL380 Gen9 and bge-m3:latest model.
    """
    
    # Ollama connection settings
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest"))
    ollama_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "30.0")))
    
    # Batch processing settings
    batch_size: int = field(default_factory=lambda: int(os.getenv("OLLAMA_EMBEDDING_BATCH_SIZE", "32")))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("OLLAMA_MAX_BATCH_SIZE", "64")))
    min_batch_size: int = field(default_factory=lambda: int(os.getenv("OLLAMA_MIN_BATCH_SIZE", "1")))
    batch_timeout: int = field(default_factory=lambda: int(os.getenv("OLLAMA_BATCH_TIMEOUT", "30000")))  # milliseconds
    processing_interval: float = field(default_factory=lambda: float(os.getenv("BATCH_PROCESSING_INTERVAL", "100")) / 1000.0)  # Convert ms to seconds
    
    # Redis configuration
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_EMBEDDING_DB", "2")))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    redis_socket_timeout: float = field(default_factory=lambda: float(os.getenv("REDIS_SOCKET_TIMEOUT", "30.0")))
    redis_connection_pool_max: int = field(default_factory=lambda: int(os.getenv("REDIS_CONNECTION_POOL_MAX", "50")))
    
    # Caching settings
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_CACHE_TTL", "3600")))  # 1 hour
    cache_enabled: bool = field(default_factory=lambda: os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true")
    cache_max_memory_mb: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_MEMORY_MB", "500")))
    
    # Error handling and retry settings
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_RETRIES", "3")))
    backoff_multiplier: float = field(default_factory=lambda: float(os.getenv("RETRY_BACKOFF_MULTIPLIER", "2.0")))
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")))
    circuit_breaker_recovery_timeout: float = field(default_factory=lambda: float(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60.0")))
    
    # Performance and monitoring
    enable_metrics: bool = field(default_factory=lambda: os.getenv("ENABLE_EMBEDDING_METRICS", "true").lower() == "true")
    metrics_collection_interval: float = field(default_factory=lambda: float(os.getenv("METRICS_COLLECTION_INTERVAL", "60.0")))
    
    # Queue management
    max_queue_size: int = field(default_factory=lambda: int(os.getenv("MAX_QUEUE_SIZE", "1000")))
    queue_warning_threshold: int = field(default_factory=lambda: int(os.getenv("QUEUE_WARNING_THRESHOLD", "800")))
    
    # Model-specific settings
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._log_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate batch sizes
        if self.batch_size > self.max_batch_size:
            logger.warning(f"batch_size ({self.batch_size}) > max_batch_size ({self.max_batch_size}), adjusting")
            self.batch_size = self.max_batch_size
        
        if self.batch_size < self.min_batch_size:
            logger.warning(f"batch_size ({self.batch_size}) < min_batch_size ({self.min_batch_size}), adjusting")
            self.batch_size = self.min_batch_size
        
        # Validate timeout values
        if self.batch_timeout < 1000:  # Less than 1 second
            logger.warning(f"batch_timeout ({self.batch_timeout}ms) is very low, consider increasing")
        
        if self.processing_interval < 0.01:  # Less than 10ms
            logger.warning(f"processing_interval ({self.processing_interval}s) is very low, may cause high CPU usage")
        
        # Validate cache settings
        if self.cache_ttl < 60:  # Less than 1 minute
            logger.warning(f"cache_ttl ({self.cache_ttl}s) is very low, cache may not be effective")
        
        # Validate retry settings
        if self.max_retries > 10:
            logger.warning(f"max_retries ({self.max_retries}) is very high, may cause long delays")
        
        # Validate queue settings
        if self.max_queue_size < self.batch_size:
            logger.error(f"max_queue_size ({self.max_queue_size}) < batch_size ({self.batch_size})")
            raise ValueError("max_queue_size must be >= batch_size")
    
    def _log_config(self):
        """Log current configuration."""
        logger.info("Ollama Batching Configuration:")
        logger.info(f"  Ollama: {self.ollama_base_url} (model: {self.ollama_model})")
        logger.info(f"  Batch size: {self.batch_size} (range: {self.min_batch_size}-{self.max_batch_size})")
        logger.info(f"  Timeout: {self.batch_timeout}ms")
        logger.info(f"  Processing interval: {self.processing_interval}s")
        logger.info(f"  Redis: {self.redis_host}:{self.redis_port}/db{self.redis_db}")
        logger.info(f"  Cache: {'enabled' if self.cache_enabled else 'disabled'} (TTL: {self.cache_ttl}s)")
        logger.info(f"  Max retries: {self.max_retries}")
        logger.info(f"  Queue size limit: {self.max_queue_size}")
    
    def to_ollama_config(self) -> Dict[str, Any]:
        """Convert to Ollama-specific configuration dictionary."""
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "timeout": self.ollama_timeout,
            "embedding_dim": self.embedding_dim,
            "max_batch_size": self.max_batch_size,
            "min_batch_size": self.min_batch_size
        }
    
    def to_redis_config(self) -> Dict[str, Any]:
        """Convert to Redis-specific configuration dictionary."""
        config = {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db,
            "socket_timeout": self.redis_socket_timeout,
            "connection_pool_max_connections": self.redis_connection_pool_max,
            "decode_responses": True
        }
        
        if self.redis_password:
            config["password"] = self.redis_password
        
        return config
    
    def get_optimized_config_for_model(self, model_name: str) -> "BatchingConfig":
        """
        Get optimized configuration for specific model.
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            New BatchingConfig instance with model-specific optimizations
        """
        # Model-specific optimizations based on testing
        model_optimizations = {
            "bge-m3:latest": {
                "batch_size": 16,
                "max_batch_size": 32,
                "processing_interval": 0.1,
                "embedding_dim": 1024
            },
            "bge-large:latest": {
                "batch_size": 8,
                "max_batch_size": 16,
                "processing_interval": 0.15,
                "embedding_dim": 1024
            },
            "all-minilm:latest": {
                "batch_size": 32,
                "max_batch_size": 64,
                "processing_interval": 0.05,
                "embedding_dim": 384
            },
            "nomic-embed-text:latest": {
                "batch_size": 16,
                "max_batch_size": 32,
                "processing_interval": 0.1,
                "embedding_dim": 768
            }
        }
        
        # Create new config with optimizations
        optimizations = model_optimizations.get(model_name, {})
        
        # Create a copy with model-specific settings
        new_config = BatchingConfig()
        new_config.ollama_model = model_name
        
        # Apply optimizations
        for key, value in optimizations.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        
        logger.info(f"Applied optimizations for model {model_name}: {optimizations}")
        return new_config
    
    @classmethod
    def from_env(cls) -> "BatchingConfig":
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BatchingConfig":
        """Create configuration from dictionary."""
        # Filter out keys that aren't part of the dataclass
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**filtered_dict)


def get_production_config() -> BatchingConfig:
    """
    Get production-ready configuration optimized for HP DL380 Gen9.
    
    Returns:
        BatchingConfig with production-optimized settings
    """
    config = BatchingConfig.from_env()
    
    # Production-specific adjustments
    if config.ollama_model == "bge-m3:latest":
        # Optimized for HP DL380 Gen9 with 32GB RAM
        config.batch_size = 16  # Optimal for bge-m3
        config.max_batch_size = 32
        config.processing_interval = 0.1
        config.cache_ttl = 3600  # 1 hour cache
        config.max_queue_size = 1000
        
        logger.info("Applied production optimizations for bge-m3:latest on HP DL380 Gen9")
    
    return config


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if environment is properly configured
    """
    required_vars = [
        "OLLAMA_BASE_URL",
        "REDIS_HOST"
    ]
    
    optional_vars = [
        "OLLAMA_EMBEDDING_BATCH_SIZE",
        "OLLAMA_BATCH_TIMEOUT",
        "REDIS_EMBEDDING_DB",
        "EMBEDDING_CACHE_TTL"
    ]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        return False
    
    # Log optional variables status
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            logger.debug(f"Optional env var {var} = {value}")
        else:
            logger.debug(f"Optional env var {var} not set, using default")
    
    logger.info("Environment validation passed")
    return True