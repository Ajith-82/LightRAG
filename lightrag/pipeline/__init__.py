"""
LightRAG Pipeline enhancements with Ollama batching optimization.

Provides enhanced embedding pipelines with intelligent batching
for significant performance improvements.
"""

from .enhanced_embedding import EnhancedEmbeddingPipeline, LegacyEmbeddingAdapter, create_enhanced_embedding_function
from .batch_integration import LightRAGBatchIntegration, enable_lightrag_batching, is_batching_available, get_recommended_config

__all__ = [
    "EnhancedEmbeddingPipeline",
    "LegacyEmbeddingAdapter", 
    "create_enhanced_embedding_function",
    "LightRAGBatchIntegration",
    "enable_lightrag_batching",
    "is_batching_available",
    "get_recommended_config"
]