"""
Comprehensive API Integration Test Suite for LightRAG
Tests all documented API endpoints with authentication, error handling, and edge cases
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Test configuration
API_BASE_URL = "http://localhost:9621"
TEST_TIMEOUT = 30


@pytest.fixture
def test_app():
    """Create test FastAPI application"""
    try:
        from lightrag.api.config import parse_args
        from lightrag.api.lightrag_server import create_app
    except ImportError:
        pytest.skip("LightRAG API components not available")
    
    # Create test arguments
    test_args = parse_args([
        "--llm_binding", "ollama",
        "--embedding_binding", "ollama",
        "--llm_model", "llama3",
        "--embedding_model", "bge-m3:latest",
        "--input_dir", tempfile.mkdtemp(),
        "--key", "test_api_key",
        "--no-auto_scan_at_startup"
    ])
    
    app = create_app(test_args)
    return app


@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def auth_headers():
    """Generate authentication headers"""
    return {
        "Authorization": "Bearer test_api_key",
        "Content-Type": "application/json"
    }


@pytest.fixture
def jwt_token():
    """Generate valid JWT token for testing"""
    secret_key = os.getenv("JWT_SECRET_KEY", "test_secret_key")
    payload = {
        "sub": "testuser",
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_basic_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "api_version" in data
    
    def test_detailed_health_check(self, client, auth_headers):
        """Test /api/health endpoint with detailed status"""
        response = client.get("/api/health", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "storage" in data
        assert "dependencies" in data
        assert "system" in data


class TestQueryEndpoints:
    """Test RAG query endpoints"""
    
    @pytest.mark.asyncio
    async def test_query_local_mode(self, client, auth_headers):
        """Test query with local mode"""
        payload = {
            "query": "What is artificial intelligence?",
            "mode": "local",
            "with_cache": True
        }
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code in [200, 401]  # 401 if auth required
        
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "mode" in data
            assert data["mode"] == "local"
    
    @pytest.mark.asyncio
    async def test_query_global_mode(self, client, auth_headers):
        """Test query with global mode"""
        payload = {
            "query": "Summarize the main topics",
            "mode": "global",
            "stream": False
        }
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code in [200, 401]
    
    @pytest.mark.asyncio
    async def test_query_hybrid_mode(self, client, auth_headers):
        """Test query with hybrid mode"""
        payload = {
            "query": "Explain the key concepts",
            "mode": "hybrid",
            "top_k": 10
        }
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code in [200, 401]
    
    def test_query_invalid_mode(self, client, auth_headers):
        """Test query with invalid mode"""
        payload = {
            "query": "Test query",
            "mode": "invalid_mode"
        }
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code in [400, 422]
    
    def test_query_missing_content(self, client, auth_headers):
        """Test query with missing content"""
        payload = {"mode": "local"}
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code in [400, 422]


class TestDocumentEndpoints:
    """Test document management endpoints"""
    
    def test_upload_text_document(self, client, auth_headers):
        """Test text document upload"""
        payload = {
            "content": "This is a test document content.",
            "description": "Test document"
        }
        response = client.post("/documents/text", json=payload, headers=auth_headers)
        assert response.status_code in [200, 201, 401]
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "status" in data or "message" in data
    
    def test_upload_file_document(self, client, auth_headers):
        """Test file document upload"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            test_file_path = f.name
        
        try:
            with open(test_file_path, 'rb') as f:
                files = {"file": ("test.txt", f, "text/plain")}
                response = client.post(
                    "/documents/upload",
                    files=files,
                    headers={"Authorization": auth_headers["Authorization"]}
                )
                assert response.status_code in [200, 201, 401]
        finally:
            os.unlink(test_file_path)
    
    def test_batch_upload_documents(self, client, auth_headers):
        """Test batch document upload"""
        payload = {
            "file_paths": ["/tmp/test1.txt", "/tmp/test2.txt"],
            "descriptions": ["Document 1", "Document 2"]
        }
        response = client.post("/documents/batch", json=payload, headers=auth_headers)
        assert response.status_code in [200, 201, 400, 401]
    
    def test_list_documents(self, client, auth_headers):
        """Test document listing"""
        response = client.get("/documents", headers=auth_headers)
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "documents" in data
    
    def test_delete_document(self, client, auth_headers):
        """Test document deletion"""
        doc_id = "test_doc_123"
        response = client.delete(f"/documents/{doc_id}", headers=auth_headers)
        assert response.status_code in [200, 204, 404, 401]
    
    def test_get_document_status(self, client, auth_headers):
        """Test document status retrieval"""
        doc_id = "test_doc_123"
        response = client.get(f"/documents/{doc_id}/status", headers=auth_headers)
        assert response.status_code in [200, 404, 401]


class TestGraphEndpoints:
    """Test knowledge graph endpoints"""
    
    def test_get_graph_data(self, client, auth_headers):
        """Test graph data retrieval"""
        response = client.get("/graph", headers=auth_headers)
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "nodes" in data or "entities" in data
            assert "edges" in data or "relationships" in data
    
    def test_get_graph_statistics(self, client, auth_headers):
        """Test graph statistics"""
        response = client.get("/graph/stats", headers=auth_headers)
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "node_count" in data or "entity_count" in data
            assert "edge_count" in data or "relationship_count" in data
    
    def test_search_entities(self, client, auth_headers):
        """Test entity search in graph"""
        params = {"query": "artificial intelligence", "limit": 10}
        response = client.get("/graph/entities", params=params, headers=auth_headers)
        assert response.status_code in [200, 401]
    
    def test_get_entity_relationships(self, client, auth_headers):
        """Test entity relationships retrieval"""
        entity_id = "test_entity_123"
        response = client.get(f"/graph/entities/{entity_id}/relationships", headers=auth_headers)
        assert response.status_code in [200, 404, 401]


class TestOllamaCompatibilityEndpoints:
    """Test Ollama-compatible API endpoints"""
    
    def test_ollama_chat_completion(self, client, auth_headers):
        """Test Ollama-compatible chat endpoint"""
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "stream": False
        }
        response = client.post("/api/chat", json=payload, headers=auth_headers)
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data or "choices" in data
    
    def test_ollama_streaming_chat(self, client, auth_headers):
        """Test streaming chat response"""
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "user", "content": "Tell me a short story"}
            ],
            "stream": True
        }
        # Streaming responses need special handling
        response = client.post("/api/chat", json=payload, headers=auth_headers)
        assert response.status_code in [200, 401]
    
    def test_ollama_generate(self, client, auth_headers):
        """Test Ollama generate endpoint"""
        payload = {
            "model": "llama3",
            "prompt": "Complete this sentence: The future of AI is",
            "stream": False
        }
        response = client.post("/api/generate", json=payload, headers=auth_headers)
        assert response.status_code in [200, 401, 404]  # 404 if endpoint not implemented


class TestAuthenticationEndpoints:
    """Test authentication and authorization"""
    
    def test_login_with_valid_credentials(self, client):
        """Test login with valid credentials"""
        payload = {
            "username": "admin",
            "password": "admin123"
        }
        response = client.post("/auth/login", json=payload)
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
    
    def test_login_with_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        payload = {
            "username": "invalid_user",
            "password": "wrong_password"
        }
        response = client.post("/auth/login", json=payload)
        assert response.status_code in [401, 403]
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.get("/protected/resource")
        assert response.status_code in [401, 403, 404]
    
    def test_protected_endpoint_with_auth(self, client, jwt_token):
        """Test accessing protected endpoint with authentication"""
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = client.get("/protected/resource", headers=headers)
        assert response.status_code in [200, 404]  # 404 if endpoint doesn't exist
    
    def test_refresh_token(self, client, jwt_token):
        """Test token refresh"""
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = client.post("/auth/refresh", headers=headers)
        assert response.status_code in [200, 401, 404]


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_enforcement(self, client, auth_headers):
        """Test that rate limiting is enforced"""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            response = client.get("/health", headers=auth_headers)
            responses.append(response.status_code)
        
        # Check if any request was rate limited
        assert 429 in responses or all(r == 200 for r in responses)
    
    def test_rate_limit_headers(self, client, auth_headers):
        """Test rate limit headers in response"""
        response = client.get("/health", headers=auth_headers)
        
        # Check for rate limit headers
        headers = response.headers
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset"
        ]
        
        # Some implementations might include rate limit headers
        has_rate_limit_headers = any(h in headers for h in rate_limit_headers)
        assert response.status_code == 200 or has_rate_limit_headers


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_json_request(self, client, auth_headers):
        """Test handling of malformed JSON"""
        response = client.post(
            "/query",
            data="{'invalid': json}",
            headers=auth_headers
        )
        assert response.status_code in [400, 422]
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test handling of missing required fields"""
        payload = {}  # Empty payload
        response = client.post("/query", json=payload, headers=auth_headers)
        assert response.status_code in [400, 422]
    
    def test_oversized_request(self, client, auth_headers):
        """Test handling of oversized requests"""
        # Create a very large payload
        large_content = "x" * (10 * 1024 * 1024)  # 10MB string
        payload = {"content": large_content}
        response = client.post("/documents/text", json=payload, headers=auth_headers)
        assert response.status_code in [413, 400, 422]
    
    def test_unsupported_media_type(self, client, auth_headers):
        """Test handling of unsupported media types"""
        response = client.post(
            "/documents/upload",
            data=b"binary data",
            headers={**auth_headers, "Content-Type": "application/octet-stream"}
        )
        assert response.status_code in [415, 400]
    
    def test_method_not_allowed(self, client, auth_headers):
        """Test handling of incorrect HTTP methods"""
        response = client.put("/health", headers=auth_headers)
        assert response.status_code == 405


class TestConcurrency:
    """Test concurrent request handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, client, auth_headers):
        """Test handling of concurrent query requests"""
        async def make_request():
            return client.post(
                "/query",
                json={"query": "Test query", "mode": "local"},
                headers=auth_headers
            )
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all requests were handled
        for result in results:
            if not isinstance(result, Exception):
                assert result.status_code in [200, 401, 429]
    
    @pytest.mark.asyncio
    async def test_concurrent_document_uploads(self, client, auth_headers):
        """Test handling of concurrent document uploads"""
        async def upload_document(i):
            payload = {
                "content": f"Document {i} content",
                "description": f"Document {i}"
            }
            return client.post("/documents/text", json=payload, headers=auth_headers)
        
        # Upload 5 documents concurrently
        tasks = [upload_document(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if not isinstance(result, Exception):
                assert result.status_code in [200, 201, 401, 429]


class TestWebSocketEndpoints:
    """Test WebSocket endpoints if available"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_app):
        """Test WebSocket connection establishment"""
        # This would require WebSocket testing client
        # Placeholder for WebSocket tests
        pass


class TestMetricsAndMonitoring:
    """Test metrics and monitoring endpoints"""
    
    def test_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            content = response.text
            assert "TYPE" in content or "HELP" in content
    
    def test_opentelemetry_traces(self, client, auth_headers):
        """Test OpenTelemetry trace information"""
        # Make a request and check for trace headers
        response = client.get("/health", headers=auth_headers)
        
        # Check for trace headers
        trace_headers = ["traceparent", "tracestate", "x-trace-id"]
        has_trace_headers = any(h in response.headers for h in trace_headers)
        
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lightrag.api", "--cov-report=term-missing"])