# LightRAG Developer Guide

## Overview

This guide covers everything developers need to know to work with, extend, and contribute to LightRAG. From setting up a development environment to implementing custom features, this documentation provides comprehensive guidance for both new and experienced developers.

## Development Environment Setup

### Prerequisites

**System Requirements:**
- Python 3.10+ (3.12 recommended)
- Node.js 18+ (for Web UI development)
- Docker 20.10+ and Docker Compose 2.0+
- Git 2.30+
- 8GB RAM minimum, 16GB recommended
- 20GB available disk space

**Development Tools:**
- VS Code or PyCharm (recommended IDEs)
- pyenv for Python version management
- bun or npm for JavaScript package management

### Initial Setup

**1. Clone and Setup Repository:**
```bash
# Clone repository
git clone <repository-url>
cd LightRAG

# Setup Python environment
pyenv install 3.12.10
pyenv local 3.12.10
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[api,test,dev]"
```

**2. Environment Configuration:**
```bash
# Copy example environment
cp env.example .env

# Configure development settings
cat > .env << EOF
# Development LLM (using Ollama for offline development)
LLM_BINDING=ollama
LLM_MODEL=llama2:7b
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-text

# Lightweight development storage
KV_STORAGE=json
VECTOR_STORAGE=nano
GRAPH_STORAGE=networkx
DOC_STATUS_STORAGE=json

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
AUTH_ENABLED=false
WORKERS=1
PORT=9621
EOF
```

**3. Start Development Services:**
```bash
# Start with development compose
docker compose up -d

# Or run API server directly
lightrag-server

# Verify setup
curl http://localhost:9621/health
```

### IDE Configuration

**VS Code Setup (.vscode/settings.json):**
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

**PyCharm Configuration:**
- Set Python interpreter to `.venv/bin/python`
- Enable pytest as test runner
- Configure Ruff for linting
- Set up run configurations for API server

## Project Structure Deep Dive

### Core Components

```
LightRAG/
├── lightrag/                    # Main library
│   ├── __init__.py             # Public API exports
│   ├── lightrag.py             # Core LightRAG class
│   ├── operate.py              # Document operations
│   ├── api/                    # FastAPI web server
│   │   ├── app.py             # Main FastAPI app
│   │   ├── routers/           # API endpoints
│   │   ├── auth/              # Authentication
│   │   └── middleware/        # Custom middleware
│   ├── kg/                     # Storage backends
│   │   ├── base.py            # Abstract base classes
│   │   ├── json_kv_impl.py    # JSON KV storage
│   │   ├── postgres_impl.py   # PostgreSQL storage
│   │   └── ...                # Other storage backends
│   ├── llm/                    # LLM integrations
│   │   ├── base.py            # Abstract LLM class
│   │   ├── openai_impl.py     # OpenAI implementation
│   │   ├── ollama_impl.py     # Ollama implementation
│   │   └── ...                # Other LLM providers
│   └── utils/                  # Utility functions
├── lightrag_mcp/               # MCP server
├── lightrag_webui/             # React frontend
├── tests/                      # Test suites
├── examples/                   # Example scripts
├── docs/                       # Documentation
└── scripts/                    # Automation scripts
```

### Key Design Patterns

**1. Dependency Injection:**
```python
# Storage backends are injected into LightRAG
rag = LightRAG(
    kv_storage=PGKVStorage(...),
    vector_storage=PGVectorStorage(...),
    graph_storage=PGGraphStorage(...),
    llm_model=OpenAILLM(...)
)
```

**2. Async/Await Pattern:**
```python
# All I/O operations are async
async def process_document(self, content: str) -> ProcessResult:
    # Initialize storages first
    await self.initialize_storages()
    
    # Process content
    result = await self._async_process(content)
    
    # Cleanup
    await self.finalize_storages()
    return result
```

**3. Plugin Architecture:**
```python
# Storage backends implement standard interfaces
class CustomStorage(BaseKVStorage):
    async def get(self, key: str) -> Any:
        # Custom implementation
        pass
```

## Development Workflow

### Code Quality Standards

**Pre-commit Hooks:**
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Code Formatting:**
```bash
# Format code with Ruff
ruff format lightrag/ tests/

# Sort imports
ruff --select I --fix lightrag/ tests/

# Lint code
ruff check lightrag/ tests/
```

**Type Checking:**
```bash
# Run mypy (if installed)
mypy lightrag/
```

### Testing

**Test Structure:**
```
tests/
├── unit/                    # Unit tests
│   ├── test_lightrag.py    # Core functionality
│   ├── test_storage.py     # Storage backends
│   └── test_llm.py         # LLM providers
├── integration/             # Integration tests
│   ├── test_api.py         # API endpoints
│   └── test_e2e.py         # End-to-end tests
├── performance/             # Performance tests
└── conftest.py             # Pytest configuration
```

**Running Tests:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_lightrag.py

# Run with coverage
pytest --cov=lightrag --cov-report=html

# Run specific test category
pytest -m "unit"
pytest -m "integration"
pytest -m "slow"

# Parallel testing
pytest -n auto
```

**Writing Tests:**
```python
import pytest
from lightrag import LightRAG
from lightrag.kg.json_kv_impl import JsonKVStorage

@pytest.mark.asyncio
async def test_document_processing():
    """Test basic document processing."""
    rag = LightRAG(
        working_dir="./test_storage",
        kv_storage=JsonKVStorage("./test_storage/kv.json"),
        llm_model=MockLLM()  # Use mock for testing
    )
    
    await rag.initialize_storages()
    
    # Test document insertion
    result = await rag.ainsert("Test document content")
    assert result.success
    
    # Test querying
    response = await rag.aquery("test query", mode="naive")
    assert len(response) > 0
    
    await rag.finalize_storages()
```

### Debugging

**Debug Configuration:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode in environment
DEBUG=true
LOG_LEVEL=DEBUG
```

**Common Debug Scenarios:**
```python
# Debug storage issues
import asyncio
from lightrag.kg.postgres_impl import PGKVStorage

async def debug_storage():
    storage = PGKVStorage(connection_string="postgresql://...")
    await storage.initialize()
    
    # Test basic operations
    await storage.set("test_key", "test_value")
    value = await storage.get("test_key")
    print(f"Retrieved: {value}")

# Debug LLM issues
from lightrag.llm.openai_impl import OpenAILLM

async def debug_llm():
    llm = OpenAILLM(api_key="sk-...")
    
    # Test generation
    response = await llm.agenerate(
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"LLM Response: {response}")
```

## Implementing Custom Features

### Custom Storage Backend

**1. Implement Base Interface:**
```python
from lightrag.kg.base import BaseKVStorage
from typing import Any, Optional

class RedisKVStorage(BaseKVStorage):
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        import aioredis
        self.client = await aioredis.from_url(self.redis_url)
    
    async def finalize(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        value = await self.client.get(key)
        if value:
            import json
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set key-value pair."""
        import json
        await self.client.set(key, json.dumps(value))
    
    async def delete(self, key: str) -> None:
        """Delete key."""
        await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.client.exists(key)
```

**2. Register Storage Backend:**
```python
# In your application
from lightrag import LightRAG

rag = LightRAG(
    kv_storage=RedisKVStorage("redis://localhost:6379"),
    # ... other configuration
)
```

### Custom LLM Provider

**1. Implement LLM Interface:**
```python
from lightrag.llm.base import BaseLLM
from typing import List, Dict, Any

class CustomLLM(BaseLLM):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    async def agenerate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """Generate text completion."""
        # Implement your custom LLM API call
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"messages": messages, **kwargs}
            ) as response:
                result = await response.json()
                return result["content"]
    
    async def aembedding(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        # Implement embedding generation
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"texts": texts}
            ) as response:
                result = await response.json()
                return result["embeddings"]
```

**2. Use Custom LLM:**
```python
rag = LightRAG(
    llm_model=CustomLLM(
        api_key="your-api-key",
        base_url="https://api.your-llm-provider.com"
    )
)
```

### Custom API Endpoints

**1. Create Custom Router:**
```python
# lightrag/api/routers/custom.py
from fastapi import APIRouter, Depends
from lightrag.api.auth import get_current_user

router = APIRouter(prefix="/custom", tags=["custom"])

@router.get("/analytics")
async def get_analytics(user=Depends(get_current_user)):
    """Custom analytics endpoint."""
    # Implement your custom logic
    return {
        "total_documents": 100,
        "total_queries": 500,
        "avg_response_time": 1.2
    }

@router.post("/batch-process")
async def batch_process(
    documents: List[str],
    user=Depends(get_current_user)
):
    """Batch process multiple documents."""
    # Implement batch processing logic
    results = []
    for doc in documents:
        # Process each document
        result = await process_document(doc)
        results.append(result)
    
    return {"processed": len(results), "results": results}
```

**2. Register Router:**
```python
# lightrag/api/app.py
from lightrag.api.routers import custom

app.include_router(custom.router)
```

### Custom Processing Pipeline

**1. Extend Operation Handler:**
```python
from lightrag.operate import BaseOperator
from typing import Any, Dict

class CustomOperator(BaseOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize custom components
    
    async def custom_preprocess(self, content: str) -> str:
        """Custom preprocessing step."""
        # Implement custom preprocessing
        # e.g., specialized text cleaning, format conversion
        processed_content = content.strip().lower()
        return processed_content
    
    async def custom_extract_entities(self, content: str) -> List[Dict]:
        """Custom entity extraction."""
        # Implement custom entity extraction logic
        # Could use specialized NLP models
        entities = []
        # ... custom extraction logic
        return entities
    
    async def process_document(self, content: str) -> Dict[str, Any]:
        """Override document processing with custom steps."""
        # Custom preprocessing
        content = await self.custom_preprocess(content)
        
        # Standard processing
        result = await super().process_document(content)
        
        # Custom post-processing
        result["custom_entities"] = await self.custom_extract_entities(content)
        
        return result
```

## Performance Optimization

### Profiling and Monitoring

**1. Performance Profiling:**
```python
import asyncio
import time
from functools import wraps

def profile_async(func):
    """Decorator to profile async functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Usage
@profile_async
async def process_large_document(content: str):
    # Processing logic
    pass
```

**2. Memory Monitoring:**
```python
import psutil
import os

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent()
    }

# Monitor during processing
print(f"Memory usage: {get_memory_usage()}")
```

### Optimization Strategies

**1. Batch Processing:**
```python
async def batch_embeddings(self, texts: List[str], batch_size: int = 32):
    """Process embeddings in batches for better performance."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await self.llm_model.aembedding(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

**2. Connection Pooling:**
```python
from asyncpg import create_pool

class OptimizedPGStorage:
    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool = None
    
    async def initialize(self):
        self.pool = await create_pool(
            self.connection_string,
            min_size=1,
            max_size=self.pool_size
        )
    
    async def execute_query(self, query: str, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
```

**3. Caching Strategies:**
```python
from functools import lru_cache
import hashlib

class CachedLLM:
    def __init__(self, llm_model, cache_size: int = 1000):
        self.llm_model = llm_model
        self.cache = {}
        self.cache_size = cache_size
    
    def _get_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key from messages."""
        content = str(messages)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def agenerate(self, messages: List[Dict], **kwargs) -> str:
        cache_key = self._get_cache_key(messages)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate response
        response = await self.llm_model.agenerate(messages, **kwargs)
        
        # Cache response (with size limit)
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = response
        
        return response
```

## Testing Guidelines

### Unit Testing Best Practices

**1. Mock External Dependencies:**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = AsyncMock()
    mock.agenerate.return_value = "Mocked response"
    mock.aembedding.return_value = [[0.1, 0.2, 0.3]]
    return mock

@pytest.mark.asyncio
async def test_document_processing(mock_llm):
    """Test document processing with mocked LLM."""
    rag = LightRAG(llm_model=mock_llm)
    
    result = await rag.ainsert("Test document")
    
    # Verify LLM was called
    mock_llm.agenerate.assert_called()
    assert result.success
```

**2. Test Data Fixtures:**
```python
@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "This is a test document about machine learning.",
        "Another document discussing artificial intelligence.",
        "A third document covering deep learning concepts."
    ]

@pytest.fixture
async def initialized_rag():
    """Pre-initialized RAG instance for testing."""
    rag = LightRAG(working_dir="./test_data")
    await rag.initialize_storages()
    yield rag
    await rag.finalize_storages()
```

### Integration Testing

**1. Database Testing:**
```python
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

@pytest_asyncio.fixture
async def postgres_container():
    """PostgreSQL container for integration testing."""
    with PostgresContainer("postgres:16") as postgres:
        connection_string = postgres.get_connection_url()
        yield connection_string

@pytest.mark.asyncio
async def test_postgres_storage(postgres_container):
    """Test PostgreSQL storage backend."""
    storage = PGKVStorage(postgres_container)
    await storage.initialize()
    
    # Test operations
    await storage.set("test_key", {"data": "test"})
    result = await storage.get("test_key")
    
    assert result["data"] == "test"
    
    await storage.finalize()
```

**2. API Testing:**
```python
from fastapi.testclient import TestClient
from lightrag.api.app import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_query_endpoint():
    """Test query endpoint."""
    response = client.post(
        "/query",
        json={"query": "test query", "mode": "naive"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```

## Contribution Guidelines

### Git Workflow

**1. Branch Naming:**
```bash
# Feature branches
git checkout -b feature/add-new-storage-backend

# Bug fixes
git checkout -b fix/resolve-memory-leak

# Documentation
git checkout -b docs/update-api-documentation
```

**2. Commit Messages:**
```bash
# Format: type(scope): description
git commit -m "feat(storage): add Redis storage backend"
git commit -m "fix(api): resolve authentication middleware issue"
git commit -m "docs(deployment): update Kubernetes guide"
```

**3. Pull Request Process:**
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit PR with clear description
6. Address review feedback
7. Merge after approval

### Code Review Checklist

**Before Submitting PR:**
- [ ] All tests pass
- [ ] Code coverage maintained
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Error handling implemented
- [ ] Performance impact considered
- [ ] Security implications reviewed

**Review Criteria:**
- Code quality and maintainability
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations
- API compatibility

### Release Process

**1. Version Management:**
```bash
# Update version in pyproject.toml
# Create release notes
# Tag release
git tag -a v1.5.0 -m "Release version 1.5.0"
git push origin v1.5.0
```

**2. Release Checklist:**
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Migration guide created (if needed)
- [ ] Performance benchmarks run
- [ ] Security scan completed
- [ ] Release notes written

## Debugging Common Issues

### Storage Issues

**Connection Problems:**
```python
# Debug storage connection
import asyncio

async def debug_storage():
    try:
        storage = PGKVStorage("postgresql://...")
        await storage.initialize()
        print("Storage connection successful")
        
        # Test basic operations
        await storage.set("test", "value")
        result = await storage.get("test")
        print(f"Test successful: {result}")
        
    except Exception as e:
        print(f"Storage error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(debug_storage())
```

### LLM Issues

**Provider Debugging:**
```python
# Debug LLM provider
async def debug_llm():
    try:
        llm = OpenAILLM(api_key="sk-...")
        
        # Test simple generation
        response = await llm.agenerate([
            {"role": "user", "content": "Hello"}
        ])
        print(f"LLM Response: {response}")
        
        # Test embeddings
        embeddings = await llm.aembedding(["test text"])
        print(f"Embedding dimension: {len(embeddings[0])}")
        
    except Exception as e:
        print(f"LLM error: {e}")
        # Check API key, network, etc.

asyncio.run(debug_llm())
```

### Performance Issues

**Memory Leaks:**
```python
import gc
import tracemalloc

# Start memory tracing
tracemalloc.start()

# Your code here
await process_large_dataset()

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# Force garbage collection
gc.collect()
```

This developer guide provides comprehensive coverage for working with LightRAG, from basic setup to advanced customization and optimization.