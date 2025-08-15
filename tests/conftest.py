"""
Global pytest configuration and fixtures for LightRAG test suite
"""

import pytest
import asyncio
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_working_dir():
    """Create a temporary working directory for the test session."""
    with tempfile.TemporaryDirectory(prefix="lightrag_test_") as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test_openai_key",
        "XAI_API_KEY": "test_xai_key",
        "AZURE_OPENAI_API_KEY": "test_azure_key",
        "JWT_SECRET_KEY": "test_jwt_secret",
        "LIGHTRAG_API_KEY": "test_api_key",
        "NODE_ENV": "test",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "id": "test_doc_1",
        "content": "This is a test document about artificial intelligence and machine learning.",
        "metadata": {
            "title": "AI Test Document",
            "author": "Test Author",
            "created_at": "2025-01-15T10:00:00Z",
        },
    }


@pytest.fixture
def sample_entities():
    """Sample entities for graph testing."""
    return [
        {
            "id": "ai",
            "name": "Artificial Intelligence",
            "type": "concept",
            "description": "The simulation of human intelligence in machines",
        },
        {
            "id": "ml",
            "name": "Machine Learning",
            "type": "concept",
            "description": "A subset of AI that enables systems to learn automatically",
        },
        {
            "id": "dl",
            "name": "Deep Learning",
            "type": "technique",
            "description": "ML technique based on artificial neural networks",
        },
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationships for graph testing."""
    return [
        {"source": "ai", "target": "ml", "type": "includes", "weight": 0.9},
        {"source": "ml", "target": "dl", "type": "includes", "weight": 0.8},
        {"source": "ai", "target": "dl", "type": "uses", "weight": 0.7},
    ]


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    import numpy as np

    return np.random.rand(768).tolist()


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a mock response from the language model.",
        "model": "test_model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response for testing."""
    import numpy as np

    return {
        "data": [{"embedding": np.random.rand(768).tolist(), "index": 0}],
        "model": "test_embedding_model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, model_name="test_model"):
        self.model_name = model_name
        self.call_count = 0

    async def agenerate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return f"Mock response to: {prompt[:50]}..."

    async def agenerate_stream(self, prompt: str, **kwargs):
        self.call_count += 1
        chunks = ["Mock ", "stream ", "response ", "to: ", prompt[:20], "..."]
        for chunk in chunks:
            yield chunk


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, model_name="test_embedding_model", dimension=768):
        self.model_name = model_name
        self.dimension = dimension
        self.call_count = 0

    async def agenerate(self, texts: List[str]) -> List[List[float]]:
        import numpy as np

        self.call_count += 1
        return [np.random.rand(self.dimension).tolist() for _ in texts]


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider fixture."""
    return MockLLMProvider()


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider fixture."""
    return MockEmbeddingProvider()


@pytest.fixture
def mock_storage_config(temp_working_dir):
    """Mock storage configuration for testing."""
    return {
        "working_dir": temp_working_dir,
        "namespace": "test",
        "enable_llm_cache": True,
        "max_async": 4,
        "timeout": 60,
    }


@pytest.fixture
def mock_api_config():
    """Mock API configuration for testing."""
    return {
        "host": "127.0.0.1",
        "port": 9621,
        "workers": 1,
        "log_level": "INFO",
        "reload": False,
        "ssl": False,
        "key": "test_api_key",
        "auth_enabled": True,
        "rate_limit_enabled": True,
        "jwt_secret_key": "test_jwt_secret",
        "jwt_expire_hours": 24,
        "jwt_algorithm": "HS256",
    }


@pytest.fixture
def mock_lightrag_instance(
    mock_storage_config, mock_llm_provider, mock_embedding_provider
):
    """Mock LightRAG instance for testing."""
    with patch("lightrag.LightRAG") as MockLightRAG:
        mock_instance = MockLightRAG.return_value

        # Mock async methods
        mock_instance.initialize_storages = AsyncMock()
        mock_instance.finalize_storages = AsyncMock()
        mock_instance.ainsert = AsyncMock()
        mock_instance.aquery = AsyncMock(return_value="Mock query response")
        mock_instance.adelete = AsyncMock()

        # Mock storage backends
        mock_instance.chunk_entity_relation_graph = Mock()
        mock_instance.entities_vdb = Mock()
        mock_instance.relationships_vdb = Mock()
        mock_instance.chunks_vdb = Mock()
        mock_instance.llm_response_cache = Mock()

        # Mock configuration
        mock_instance.working_dir = mock_storage_config["working_dir"]
        mock_instance.llm_model_func = mock_llm_provider.agenerate
        mock_instance.embedding_func = mock_embedding_provider.agenerate

        yield mock_instance


@pytest.fixture
def authenticated_client_headers():
    """Headers for authenticated API client."""
    return {"Authorization": "Bearer test_api_key", "Content-Type": "application/json"}


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Add any cleanup logic here
    pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["NODE_ENV"] = "test"

    # Suppress warnings for cleaner test output
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    yield

    # Cleanup after all tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


# Custom pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "storage: mark test as storage test")
    config.addinivalue_line("markers", "auth: mark test as authentication test")
    config.addinivalue_line("markers", "rate_limit: mark test as rate limiting test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and paths."""
    for item in items:
        # Add markers based on test file path
        if "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        if "test_security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        if "test_storage" in str(item.fspath):
            item.add_marker(pytest.mark.storage)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add markers based on test names
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
        if "auth" in item.name.lower():
            item.add_marker(pytest.mark.auth)
        if "rate_limit" in item.name.lower():
            item.add_marker(pytest.mark.rate_limit)


# Custom pytest fixtures for specific test scenarios
@pytest.fixture
def mock_database_connection():
    """Mock database connection for storage tests."""
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def mock_redis_connection():
    """Mock Redis connection for cache tests."""
    with patch("redis.Redis") as mock_redis:
        mock_instance = Mock()
        mock_redis.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_vector_db():
    """Mock vector database for vector storage tests."""
    with patch("lightrag.kg.vector_storage.NanoVectorDBStorage") as mock_vdb:
        mock_instance = Mock()
        mock_instance.upsert = AsyncMock()
        mock_instance.query = AsyncMock(return_value=[])
        mock_instance.get_all_vectors = AsyncMock(return_value=[])
        mock_vdb.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# Helper functions for tests
def assert_valid_response(response_data: Dict[str, Any], required_fields: List[str]):
    """Assert that response contains required fields."""
    assert isinstance(response_data, dict)
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"


def assert_valid_entity(entity: Dict[str, Any]):
    """Assert that entity has valid structure."""
    required_fields = ["id", "name", "type"]
    assert_valid_response(entity, required_fields)


def assert_valid_relationship(relationship: Dict[str, Any]):
    """Assert that relationship has valid structure."""
    required_fields = ["source", "target", "type"]
    assert_valid_response(relationship, required_fields)


def create_test_file(content: str, suffix: str = ".txt") -> str:
    """Create a temporary test file with given content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name
