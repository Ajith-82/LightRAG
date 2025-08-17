"""
xAI configuration optimizer for LightRAG.

Provides automatic configuration adjustments and validation for optimal
xAI Grok integration, preventing timeout issues and ensuring reliability.
"""

import logging
import os
from typing import Dict, Any, Tuple
import warnings

logger = logging.getLogger(__name__)


class XAIConfigOptimizer:
    """Optimizer for xAI Grok configuration in LightRAG."""
    
    # Recommended configurations for different xAI models
    XAI_MODEL_CONFIGS = {
        "grok-3-mini": {
            "max_async": 2,
            "timeout": 240,
            "max_parallel_insert": 1,
            "embedding_func_max_async": 4,
            "description": "Fastest xAI model, good for general use"
        },
        "grok-2-1212": {
            "max_async": 1,
            "timeout": 300,
            "max_parallel_insert": 1,
            "embedding_func_max_async": 2,
            "description": "High-quality responses, slower processing"
        },
        "grok-2-vision-1212": {
            "max_async": 1,
            "timeout": 360,
            "max_parallel_insert": 1,
            "embedding_func_max_async": 2,
            "description": "Vision-enabled model, requires more resources"
        }
    }
    
    def __init__(self):
        self.current_config = self._load_current_config()
    
    def _load_current_config(self) -> Dict[str, Any]:
        """Load current environment configuration."""
        return {
            "llm_binding": os.getenv("LLM_BINDING", "").lower(),
            "llm_model": os.getenv("LLM_MODEL", "").lower(),
            "max_async": int(os.getenv("MAX_ASYNC", "4")),
            "timeout": int(os.getenv("TIMEOUT", "240")),
            "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "2")),
            "embedding_func_max_async": int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "8")),
            "embedding_binding": os.getenv("EMBEDDING_BINDING", "").lower(),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "")
        }
    
    def is_xai_configured(self) -> bool:
        """Check if xAI is currently configured."""
        return self.current_config["llm_binding"] == "xai"
    
    def get_recommended_config(self, model: str = None) -> Dict[str, Any]:
        """Get recommended configuration for xAI model."""
        if model is None:
            model = self.current_config.get("llm_model", "grok-3-mini")
        
        if not model:  # Handle empty string case
            model = "grok-3-mini"
        
        # Normalize model name
        model = model.lower().strip()
        
        # Find best match for model
        if model in self.XAI_MODEL_CONFIGS:
            base_config = self.XAI_MODEL_CONFIGS[model].copy()
        elif "grok-3" in model:
            base_config = self.XAI_MODEL_CONFIGS["grok-3-mini"].copy()
        elif "grok-2" in model and "vision" in model:
            base_config = self.XAI_MODEL_CONFIGS["grok-2-vision-1212"].copy()
        elif "grok-2" in model:
            base_config = self.XAI_MODEL_CONFIGS["grok-2-1212"].copy()
        else:
            # Default conservative settings for unknown models
            base_config = {
                "max_async": 1,
                "timeout": 300,
                "max_parallel_insert": 1,
                "embedding_func_max_async": 2,
                "description": "Conservative settings for unknown model"
            }
        
        return base_config
    
    def validate_current_config(self) -> Tuple[bool, list[str]]:
        """Validate current configuration against xAI best practices."""
        if not self.is_xai_configured():
            return True, []  # Not using xAI, no validation needed
        
        issues = []
        model = self.current_config.get("llm_model", "unknown")
        recommended = self.get_recommended_config(model)
        
        # Check MAX_ASYNC
        if self.current_config["max_async"] > recommended["max_async"]:
            issues.append(
                f"MAX_ASYNC={self.current_config['max_async']} is too high for {model}. "
                f"Recommended: {recommended['max_async']} to prevent timeout issues."
            )
        
        # Check TIMEOUT
        if self.current_config["timeout"] < recommended["timeout"]:
            issues.append(
                f"TIMEOUT={self.current_config['timeout']} may be too low for {model}. "
                f"Recommended: {recommended['timeout']} seconds."
            )
        
        # Check parallel processing
        if self.current_config["max_parallel_insert"] > recommended["max_parallel_insert"]:
            issues.append(
                f"MAX_PARALLEL_INSERT={self.current_config['max_parallel_insert']} is too high for {model}. "
                f"Recommended: {recommended['max_parallel_insert']}."
            )
        
        # Check embedding concurrency
        if self.current_config["embedding_func_max_async"] > recommended["embedding_func_max_async"]:
            issues.append(
                f"EMBEDDING_FUNC_MAX_ASYNC={self.current_config['embedding_func_max_async']} may cause Ollama timeouts. "
                f"Recommended: {recommended['embedding_func_max_async']}."
            )
        
        # Check for known problematic combinations
        if (self.current_config["embedding_binding"] == "ollama" and 
            self.current_config["max_async"] > 2):
            issues.append(
                "High MAX_ASYNC with Ollama embeddings can cause timeout issues. "
                "Consider reducing MAX_ASYNC to 2 or switching to OpenAI embeddings."
            )
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def generate_optimal_env_config(self, model: str = None) -> str:
        """Generate optimal .env configuration for xAI."""
        if model is None:
            model = self.current_config.get("llm_model", "grok-3-mini")
        
        if not model:  # Handle empty string case
            model = "grok-3-mini"
        
        recommended = self.get_recommended_config(model)
        
        config_lines = [
            "# Optimized xAI Configuration",
            f"# Model: {model} - {recommended.get('description', '')}",
            "# Generated by LightRAG xAI Optimizer",
            "",
            "# LLM Configuration",
            "LLM_BINDING=xai",
            f"LLM_MODEL={model}",
            "LLM_BINDING_HOST=https://api.x.ai/v1",
            "LLM_BINDING_API_KEY=your_xai_api_key_here",
            "",
            "# Concurrency Configuration (Optimized for xAI)",
            f"MAX_ASYNC={recommended['max_async']}",
            f"MAX_PARALLEL_INSERT={recommended['max_parallel_insert']}",
            f"EMBEDDING_FUNC_MAX_ASYNC={recommended['embedding_func_max_async']}",
            "",
            "# Timeout Configuration",
            f"TIMEOUT={recommended['timeout']}",
            "",
            "# Recommended Embedding Configuration",
            "EMBEDDING_BINDING=ollama",
            "EMBEDDING_MODEL=bge-m3:latest",
            "EMBEDDING_DIM=1024",
            "EMBEDDING_BINDING_HOST=http://localhost:11434",
            "",
            "# Alternative: Use OpenAI embeddings for better reliability",
            "# EMBEDDING_BINDING=openai",
            "# EMBEDDING_MODEL=text-embedding-3-small",
            "# EMBEDDING_DIM=1536",
            "# EMBEDDING_BINDING_API_KEY=your_openai_api_key",
        ]
        
        return "\n".join(config_lines)
    
    def auto_configure(self, apply_changes: bool = False) -> Dict[str, Any]:
        """Automatically configure environment for optimal xAI performance."""
        if not self.is_xai_configured():
            return {"status": "not_applicable", "message": "xAI not configured"}
        
        is_valid, issues = self.validate_current_config()
        
        if is_valid:
            return {
                "status": "optimal",
                "message": "Configuration is already optimal for xAI",
                "current_config": self.current_config
            }
        
        model = self.current_config.get("llm_model", "grok-3-mini")
        recommended = self.get_recommended_config(model)
        
        changes = {}
        
        # Determine what needs to be changed
        if self.current_config["max_async"] > recommended["max_async"]:
            changes["MAX_ASYNC"] = recommended["max_async"]
        
        if self.current_config["timeout"] < recommended["timeout"]:
            changes["TIMEOUT"] = recommended["timeout"]
        
        if self.current_config["max_parallel_insert"] > recommended["max_parallel_insert"]:
            changes["MAX_PARALLEL_INSERT"] = recommended["max_parallel_insert"]
        
        if self.current_config["embedding_func_max_async"] > recommended["embedding_func_max_async"]:
            changes["EMBEDDING_FUNC_MAX_ASYNC"] = recommended["embedding_func_max_async"]
        
        if apply_changes:
            # Apply changes to environment
            for key, value in changes.items():
                os.environ[key] = str(value)
                logger.info(f"Applied xAI optimization: {key}={value}")
        
        return {
            "status": "optimized" if apply_changes else "recommendations_available",
            "message": f"Configuration optimized for {model}",
            "issues": issues,
            "changes": changes,
            "recommended_config": recommended
        }
    
    def print_optimization_report(self):
        """Print detailed optimization report."""
        if not self.is_xai_configured():
            print("❌ xAI not configured - optimization not applicable")
            return
        
        print(f"🔍 xAI Configuration Analysis")
        print(f"{'='*50}")
        
        model = self.current_config.get("llm_model", "unknown")
        print(f"Model: {model}")
        
        is_valid, issues = self.validate_current_config()
        
        if is_valid:
            print("✅ Configuration is optimal for xAI")
        else:
            print(f"⚠️  Found {len(issues)} configuration issues:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        print(f"\n📊 Current Configuration:")
        print(f"   MAX_ASYNC: {self.current_config['max_async']}")
        print(f"   TIMEOUT: {self.current_config['timeout']}")
        print(f"   MAX_PARALLEL_INSERT: {self.current_config['max_parallel_insert']}")
        print(f"   EMBEDDING_FUNC_MAX_ASYNC: {self.current_config['embedding_func_max_async']}")
        
        recommended = self.get_recommended_config(model)
        print(f"\n💡 Recommended Configuration:")
        print(f"   MAX_ASYNC: {recommended['max_async']}")
        print(f"   TIMEOUT: {recommended['timeout']}")
        print(f"   MAX_PARALLEL_INSERT: {recommended['max_parallel_insert']}")
        print(f"   EMBEDDING_FUNC_MAX_ASYNC: {recommended['embedding_func_max_async']}")
        
        if not is_valid:
            print(f"\n🔧 To apply optimizations:")
            print(f"   from lightrag.xai_optimizer import optimize_xai_config")
            print(f"   optimize_xai_config(apply_changes=True)")


def optimize_xai_config(apply_changes: bool = False) -> Dict[str, Any]:
    """Convenience function to optimize xAI configuration."""
    optimizer = XAIConfigOptimizer()
    return optimizer.auto_configure(apply_changes=apply_changes)


def validate_xai_config() -> Tuple[bool, list[str]]:
    """Convenience function to validate xAI configuration."""
    optimizer = XAIConfigOptimizer()
    return optimizer.validate_current_config()


def generate_xai_env_config(model: str = "grok-3-mini") -> str:
    """Convenience function to generate optimal .env configuration."""
    optimizer = XAIConfigOptimizer()
    return optimizer.generate_optimal_env_config(model)


def print_xai_optimization_report():
    """Convenience function to print optimization report."""
    optimizer = XAIConfigOptimizer()
    optimizer.print_optimization_report()


# Automatic validation on import if xAI is configured
def _auto_validate_on_import():
    """Automatically validate configuration when module is imported."""
    try:
        optimizer = XAIConfigOptimizer()
        if optimizer.is_xai_configured():
            is_valid, issues = optimizer.validate_current_config()
            if not is_valid:
                warning_msg = (
                    f"xAI configuration may not be optimal. Found {len(issues)} issues. "
                    f"Run lightrag.xai_optimizer.print_xai_optimization_report() for details."
                )
                warnings.warn(warning_msg, UserWarning)
                logger.warning(f"xAI configuration issues detected: {issues}")
    except Exception as e:
        # Don't fail imports due to validation issues
        logger.debug(f"xAI configuration validation failed: {e}")


# Run automatic validation
_auto_validate_on_import()