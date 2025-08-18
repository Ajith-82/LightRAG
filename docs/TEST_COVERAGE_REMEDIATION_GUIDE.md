# Test Coverage Remediation Guide

## Quick Implementation Steps

### 1. Configuration Fixture Implementation (Priority 1)

Create `/opt/developments/LightRAG/tests/fixtures/lightrag_fixtures.py`:

```python
import pytest
import os
import tempfile
from pathlib import Path
from lightrag import LightRAG
from lightrag.utils import initialize_global_args

@pytest.fixture(scope="session")
async def test_global_args():
    """Initialize global arguments for test environment"""
    args = initialize_global_args({
        'working_dir': tempfile.mkdtemp(),
        'llm_binding': 'mock',
        'embedding_binding': 'mock',
        'enable_llm_cache': False,
        'enable_local_embedding': True,
    })
    yield args
    # Cleanup
    import shutil
    shutil.rmtree(args.working_dir, ignore_errors=True)

@pytest.fixture
async def mock_lightrag(test_global_args):
    """Provide fully configured LightRAG instance for testing"""
    rag = LightRAG(
        working_dir=test_global_args.working_dir,
        llm_model_func=mock_llm_func,
        embedding_func=mock_embedding_func,
    )
    await rag.initialize_storages()
    yield rag
    await rag.finalize_storages()

# Mock functions for testing
async def mock_llm_func(prompt, **kwargs):
    return "Mock LLM response"

async def mock_embedding_func(text):
    import numpy as np
    return np.random.rand(384).tolist()  # Mock embedding
```

### 2. Service Mocking (Priority 2)

Add to `/opt/developments/LightRAG/tests/fixtures/service_mocks.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_redis():
    """Mock Redis client for caching tests"""
    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.ltrim = AsyncMock(return_value=True)
    return redis_mock

@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL connection for storage tests"""
    conn_mock = MagicMock()
    cursor_mock = MagicMock()
    cursor_mock.fetchone = MagicMock(return_value=None)
    cursor_mock.fetchall = MagicMock(return_value=[])
    cursor_mock.execute = MagicMock()
    conn_mock.cursor = MagicMock(return_value=cursor_mock)
    conn_mock.commit = MagicMock()
    return conn_mock

@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver for graph storage tests"""
    driver_mock = MagicMock()
    session_mock = MagicMock()
    session_mock.run = MagicMock(return_value=[])
    driver_mock.session = MagicMock(return_value=session_mock)
    return driver_mock
```

### 3. Update Existing Test Files

Example fix for `/opt/developments/LightRAG/tests/core/test_lightrag_core.py`:

```python
# Add these imports at the top
from tests.fixtures.lightrag_fixtures import mock_lightrag, test_global_args
from tests.fixtures.service_mocks import mock_redis, mock_postgres

# Update test class
@pytest.mark.asyncio
class TestLightRAGCore:
    async def test_document_insertion(self, mock_lightrag):
        """Test document insertion with proper LightRAG instance"""
        result = await mock_lightrag.ainsert("Test document content")
        assert result is not None
        
    async def test_query_execution(self, mock_lightrag):
        """Test query execution with mock instance"""
        # Insert test data first
        await mock_lightrag.ainsert("Sample document for testing")
        
        # Test different query modes
        result = await mock_lightrag.aquery("test query", mode="local")
        assert isinstance(result, str)
```

### 4. Async Test Configuration

Update `/opt/developments/LightRAG/pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
timeout = 300  # 5 minute timeout for long tests
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

## Implementation Order

### Day 1-2: Core Configuration
1. Create fixture files above
2. Update 5-10 core test files to use fixtures
3. Verify basic test execution works
4. Expected coverage jump: 6% → 25%

### Day 3-4: Service Mocking  
1. Implement Redis/PostgreSQL mocks
2. Update storage backend tests
3. Fix integration test dependencies
4. Expected coverage jump: 25% → 45%

### Day 5-7: Full Implementation
1. Apply fixtures to all remaining test files
2. Fix async configuration issues
3. Optimize test performance
4. Expected coverage jump: 45% → 70%+

## Validation Commands

```bash
# Test configuration fixtures
pytest tests/fixtures/ -v

# Test core functionality with new fixtures
pytest tests/core/ -v --tb=short

# Check coverage improvements
pytest tests/ --cov=lightrag --cov-report=term-missing

# Run specific test categories
pytest -m "not slow" --cov=lightrag
pytest tests/auth/ -v  # Known working tests
```

## Troubleshooting Common Issues

### Issue: `global_args` still undefined
**Solution**: Ensure fixture is imported and used in test function signature

### Issue: Async tests timeout  
**Solution**: Add `@pytest.mark.asyncio` decorator and check timeout settings

### Issue: Mock services not working
**Solution**: Verify mock is injected into the actual code being tested, not just the test

### Issue: Coverage not improving
**Solution**: Check that tests are actually exercising the code, not just passing

## Quick Win Targets

Focus on these files for immediate coverage improvements:

1. **lightrag/utils.py** (8.74% → 70%)
   - Many utility functions easily testable
   - High impact on overall coverage

2. **lightrag/operate.py** (2.45% → 60%) 
   - Core RAG operations
   - Requires configuration fixture

3. **lightrag/api/middleware/rate_limiter.py** (52% → 80%)
   - Already partially working
   - Needs Redis mock

## Expected Timeline

- **Week 1**: Configuration fixes, basic mocking → 45% coverage
- **Week 2**: Full service mocking, async stabilization → 70% coverage
- **Week 3**: Optimization and validation → 70%+ sustained

This guide provides concrete implementation steps to resolve the test coverage blockers identified in the analysis report.