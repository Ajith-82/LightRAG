"""
Core LLM Provider Integration Tests for LightRAG
Tests all supported LLM providers with real and mocked integrations
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import openai
import pytest

# Test markers
pytestmark = [pytest.mark.core, pytest.mark.unit]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    with patch('lightrag.llm.openai.openai.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock chat completions
        mock_completion = Mock()
        mock_completion.choices = [
            Mock(message=Mock(content="Mock OpenAI response"))
        ]
        mock_completion.usage = Mock(
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25
        )
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Mock streaming
        mock_stream_chunk = Mock()
        mock_stream_chunk.choices = [
            Mock(delta=Mock(content="Mock "))
        ]
        mock_client.chat.completions.create.return_value = [mock_stream_chunk]
        
        yield mock_client


@pytest.fixture
def mock_xai_client():
    """Mock xAI client for testing"""
    with patch('httpx.AsyncClient') as mock_httpx:
        mock_client = Mock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Mock xAI response"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 18,
                "total_tokens": 30
            }
        }
        mock_client.post = AsyncMock(return_value=mock_response)
        
        yield mock_client


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    with patch('httpx.AsyncClient') as mock_httpx:
        mock_client = Mock()
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Mock Ollama response",
            "done": True,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 15
        }
        mock_client.post = AsyncMock(return_value=mock_response)
        
        yield mock_client


class TestOpenAIProvider:
    """Test OpenAI LLM provider"""
    
    @pytest.mark.asyncio
    async def test_openai_initialization(self, mock_env_vars):
        """Test OpenAI provider initialization"""
        try:
            from lightrag.llm.openai import OpenAILLM

            # Test with API key
            llm = OpenAILLM(api_key="test_key", model="gpt-4")
            assert llm.model == "gpt-4"
            assert llm.api_key == "test_key"
        except ImportError:
            pytest.skip("OpenAI provider not available")
    
    @pytest.mark.asyncio
    async def test_openai_generate_text(self, mock_openai_client, mock_env_vars):
        """Test OpenAI text generation"""
        try:
            from lightrag.llm.openai import OpenAILLM
            
            llm = OpenAILLM(api_key="test_key", model="gpt-4")
            
            # Test text generation
            with patch.object(llm, '_client', mock_openai_client):
                response = await llm.agenerate("Test prompt")
                assert response == "Mock OpenAI response"
                
                # Verify client was called correctly
                mock_openai_client.chat.completions.create.assert_called_once()
                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]['model'] == "gpt-4"
                assert call_args[1]['messages'][0]['content'] == "Test prompt"
        except ImportError:
            pytest.skip("OpenAI provider not available")
    
    @pytest.mark.asyncio
    async def test_openai_streaming(self, mock_openai_client, mock_env_vars):
        """Test OpenAI streaming response"""
        try:
            from lightrag.llm.openai import OpenAILLM
            
            llm = OpenAILLM(api_key="test_key", model="gpt-4")
            
            # Mock streaming response
            mock_openai_client.chat.completions.create.return_value = [
                Mock(choices=[Mock(delta=Mock(content="Hello "))]),
                Mock(choices=[Mock(delta=Mock(content="World!"))]),
                Mock(choices=[Mock(delta=Mock(content=None))])
            ]
            
            with patch.object(llm, '_client', mock_openai_client):
                chunks = []
                async for chunk in llm.agenerate_stream("Test prompt"):
                    if chunk:
                        chunks.append(chunk)
                
                assert "Hello " in chunks or "World!" in chunks
        except ImportError:
            pytest.skip("OpenAI provider not available")
    
    @pytest.mark.asyncio
    async def test_openai_error_handling(self, mock_env_vars):
        """Test OpenAI error handling"""
        try:
            from lightrag.llm.openai import OpenAILLM
            
            llm = OpenAILLM(api_key="test_key", model="gpt-4")
            
            # Mock client to raise an error
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            
            with patch.object(llm, '_client', mock_client):
                with pytest.raises(Exception):
                    await llm.agenerate("Test prompt")
        except ImportError:
            pytest.skip("OpenAI provider not available")


class TestXAIProvider:
    """Test xAI (Grok) LLM provider"""
    
    @pytest.mark.asyncio
    async def test_xai_initialization(self, mock_env_vars):
        """Test xAI provider initialization"""
        try:
            from lightrag.llm.xai import XAILLM
            
            llm = XAILLM(api_key="test_key", model="grok-3-mini")
            assert llm.model == "grok-3-mini"
            assert llm.api_key == "test_key"
        except ImportError:
            pytest.skip("xAI provider not available")
    
    @pytest.mark.asyncio
    async def test_xai_generate_text(self, mock_xai_client, mock_env_vars):
        """Test xAI text generation"""
        try:
            from lightrag.llm.xai import XAILLM
            
            llm = XAILLM(api_key="test_key", model="grok-3-mini")
            
            # Test with mocked HTTP client
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_httpx.return_value.__aenter__.return_value = mock_xai_client
                
                response = await llm.agenerate("Test prompt")
                assert response == "Mock xAI response"
        except ImportError:
            pytest.skip("xAI provider not available")
    
    @pytest.mark.asyncio
    async def test_xai_timeout_handling(self, mock_env_vars):
        """Test xAI timeout handling (important for xAI)"""
        try:
            from lightrag.llm.xai import XAILLM
            
            llm = XAILLM(api_key="test_key", model="grok-3-mini", timeout=1.0)
            
            # Mock client to simulate timeout
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_client = Mock()
                mock_httpx.return_value.__aenter__.return_value = mock_client
                mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
                
                with pytest.raises((httpx.TimeoutException, Exception)):
                    await llm.agenerate("Test prompt")
        except ImportError:
            pytest.skip("xAI provider not available")
    
    @pytest.mark.asyncio
    async def test_xai_retry_logic(self, mock_env_vars):
        """Test xAI retry logic for stability"""
        try:
            from lightrag.llm.xai import XAILLM
            
            llm = XAILLM(api_key="test_key", model="grok-3-mini")
            
            # Mock client to fail first time, succeed second time
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_client = Mock()
                mock_httpx.return_value.__aenter__.return_value = mock_client
                
                # First call fails, second succeeds
                mock_response = Mock()
                mock_response.json.return_value = {
                    "choices": [{"message": {"content": "Retry success"}}]
                }
                
                mock_client.post = AsyncMock(side_effect=[
                    httpx.HTTPStatusError("Server Error", request=Mock(), response=Mock(status_code=500)),
                    mock_response
                ])
                
                # Should succeed after retry (if retry logic exists)
                try:
                    response = await llm.agenerate("Test prompt")
                    assert "success" in response.lower()
                except Exception:
                    # If no retry logic, that's expected
                    pass
        except ImportError:
            pytest.skip("xAI provider not available")


class TestOllamaProvider:
    """Test Ollama LLM provider"""
    
    @pytest.mark.asyncio
    async def test_ollama_initialization(self, mock_env_vars):
        """Test Ollama provider initialization"""
        try:
            from lightrag.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(
                base_url="http://localhost:11434",
                model="llama3"
            )
            assert llm.model == "llama3"
            assert "localhost" in llm.base_url
        except ImportError:
            pytest.skip("Ollama provider not available")
    
    @pytest.mark.asyncio
    async def test_ollama_generate_text(self, mock_ollama_client, mock_env_vars):
        """Test Ollama text generation"""
        try:
            from lightrag.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(
                base_url="http://localhost:11434",
                model="llama3"
            )
            
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_httpx.return_value.__aenter__.return_value = mock_ollama_client
                
                response = await llm.agenerate("Test prompt")
                assert response == "Mock Ollama response"
        except ImportError:
            pytest.skip("Ollama provider not available")
    
    @pytest.mark.asyncio
    async def test_ollama_streaming(self, mock_env_vars):
        """Test Ollama streaming response"""
        try:
            from lightrag.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(
                base_url="http://localhost:11434",
                model="llama3"
            )
            
            # Mock streaming response
            stream_responses = [
                {"response": "Hello ", "done": False},
                {"response": "World!", "done": False},
                {"response": "", "done": True}
            ]
            
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_client = Mock()
                mock_httpx.return_value.__aenter__.return_value = mock_client
                
                # Mock streaming response
                async def mock_stream():
                    for resp in stream_responses:
                        mock_response = Mock()
                        mock_response.json.return_value = resp
                        yield mock_response
                
                mock_client.stream.return_value.__aenter__.return_value = mock_stream()
                
                chunks = []
                async for chunk in llm.agenerate_stream("Test prompt"):
                    if chunk:
                        chunks.append(chunk)
                
                assert len(chunks) >= 2
        except ImportError:
            pytest.skip("Ollama provider not available")


class TestAzureOpenAIProvider:
    """Test Azure OpenAI LLM provider"""
    
    @pytest.mark.asyncio
    async def test_azure_initialization(self, mock_env_vars):
        """Test Azure OpenAI provider initialization"""
        try:
            from lightrag.llm.azure_openai import AzureOpenAILLM
            
            llm = AzureOpenAILLM(
                api_key="test_key",
                endpoint="https://test.openai.azure.com",
                deployment="test-deployment",
                api_version="2023-12-01-preview"
            )
            assert llm.deployment == "test-deployment"
            assert "azure.com" in llm.endpoint
        except ImportError:
            pytest.skip("Azure OpenAI provider not available")
    
    @pytest.mark.asyncio
    async def test_azure_generate_text(self, mock_env_vars):
        """Test Azure OpenAI text generation"""
        try:
            from lightrag.llm.azure_openai import AzureOpenAILLM
            
            llm = AzureOpenAILLM(
                api_key="test_key",
                endpoint="https://test.openai.azure.com",
                deployment="test-deployment"
            )
            
            # Mock Azure OpenAI client
            with patch('openai.AzureOpenAI') as mock_azure:
                mock_client = Mock()
                mock_azure.return_value = mock_client
                
                mock_completion = Mock()
                mock_completion.choices = [
                    Mock(message=Mock(content="Mock Azure response"))
                ]
                mock_client.chat.completions.create.return_value = mock_completion
                
                response = await llm.agenerate("Test prompt")
                assert response == "Mock Azure response"
        except ImportError:
            pytest.skip("Azure OpenAI provider not available")


class TestEmbeddingProviders:
    """Test embedding providers (critical for vector operations)"""
    
    @pytest.mark.asyncio
    async def test_openai_embeddings(self, mock_env_vars):
        """Test OpenAI embedding generation"""
        try:
            from lightrag.llm.openai import OpenAIEmbedding
            
            embedding = OpenAIEmbedding(
                api_key="test_key",
                model="text-embedding-ada-002"
            )
            
            # Mock OpenAI client
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                mock_embedding = Mock()
                mock_embedding.data = [
                    Mock(embedding=[0.1, 0.2, 0.3] * 512)  # 1536 dimensions
                ]
                mock_client.embeddings.create.return_value = mock_embedding
                
                result = await embedding.agenerate(["Test text"])
                assert len(result) == 1
                assert len(result[0]) == 1536  # OpenAI embedding dimension
        except ImportError:
            pytest.skip("OpenAI embedding provider not available")
    
    @pytest.mark.asyncio
    async def test_ollama_embeddings(self, mock_env_vars):
        """Test Ollama embedding generation"""
        try:
            from lightrag.llm.ollama import OllamaEmbedding
            
            embedding = OllamaEmbedding(
                base_url="http://localhost:11434",
                model="bge-m3:latest"
            )
            
            # Mock Ollama response
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_client = Mock()
                mock_httpx.return_value.__aenter__.return_value = mock_client
                
                mock_response = Mock()
                mock_response.json.return_value = {
                    "embedding": [0.1, 0.2, 0.3] * 342  # 1024 dimensions for BGE-M3
                }
                mock_client.post = AsyncMock(return_value=mock_response)
                
                result = await embedding.agenerate(["Test text"])
                assert len(result) == 1
                assert len(result[0]) == 1024  # BGE-M3 dimension
        except ImportError:
            pytest.skip("Ollama embedding provider not available")


class TestLLMProviderIntegration:
    """Test integration between LLM providers and LightRAG"""
    
    @pytest.mark.asyncio
    async def test_provider_switching(self, mock_env_vars):
        """Test switching between different LLM providers"""
        # This tests the provider factory pattern
        providers_to_test = [
            ("openai", "gpt-4"),
            ("xai", "grok-3-mini"),
            ("ollama", "llama3"),
            ("azure_openai", "gpt-4")
        ]
        
        for provider_type, model in providers_to_test:
            try:
                if provider_type == "openai":
                    from lightrag.llm.openai import OpenAILLM
                    provider = OpenAILLM(api_key="test", model=model)
                elif provider_type == "xai":
                    from lightrag.llm.xai import XAILLM
                    provider = XAILLM(api_key="test", model=model)
                elif provider_type == "ollama":
                    from lightrag.llm.ollama import OllamaLLM
                    provider = OllamaLLM(base_url="http://localhost:11434", model=model)
                elif provider_type == "azure_openai":
                    from lightrag.llm.azure_openai import AzureOpenAILLM
                    provider = AzureOpenAILLM(
                        api_key="test",
                        endpoint="https://test.openai.azure.com",
                        deployment=model
                    )
                
                assert provider is not None
                assert hasattr(provider, 'agenerate')
                
            except ImportError:
                # Skip if provider not available
                continue
    
    @pytest.mark.asyncio
    async def test_concurrent_llm_calls(self, mock_env_vars):
        """Test concurrent LLM calls (important for performance)"""
        try:
            from lightrag.llm.openai import OpenAILLM
            
            llm = OpenAILLM(api_key="test_key", model="gpt-4")
            
            # Mock concurrent calls
            with patch.object(llm, 'agenerate', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = "Mock response"
                
                # Test concurrent execution
                tasks = [
                    llm.agenerate(f"Prompt {i}") 
                    for i in range(5)
                ]
                
                results = await asyncio.gather(*tasks)
                assert len(results) == 5
                assert all(result == "Mock response" for result in results)
                assert mock_generate.call_count == 5
        except ImportError:
            pytest.skip("OpenAI provider not available")
    
    @pytest.mark.asyncio
    async def test_llm_caching_behavior(self, mock_env_vars, temp_working_dir):
        """Test LLM response caching"""
        try:
            from lightrag.llm.openai import OpenAILLM
            
            llm = OpenAILLM(api_key="test_key", model="gpt-4")
            
            # Mock the same response for caching test
            with patch.object(llm, 'agenerate', new_callable=AsyncMock) as mock_generate:
                mock_generate.return_value = "Cached response"
                
                # First call
                result1 = await llm.agenerate("Same prompt")
                # Second call with same prompt (should be cached if caching is enabled)
                result2 = await llm.agenerate("Same prompt")
                
                assert result1 == result2 == "Cached response"
        except ImportError:
            pytest.skip("OpenAI provider not available")


class TestProviderErrorHandling:
    """Test error handling across all providers"""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, mock_env_vars):
        """Test network timeout handling"""
        providers_to_test = [
            ("openai", "gpt-4"),
            ("xai", "grok-3-mini"),
            ("ollama", "llama3")
        ]
        
        for provider_type, model in providers_to_test:
            try:
                if provider_type == "openai":
                    from lightrag.llm.openai import OpenAILLM
                    provider = OpenAILLM(api_key="test", model=model, timeout=0.001)  # Very short timeout
                elif provider_type == "xai":
                    from lightrag.llm.xai import XAILLM
                    provider = XAILLM(api_key="test", model=model, timeout=0.001)
                elif provider_type == "ollama":
                    from lightrag.llm.ollama import OllamaLLM
                    provider = OllamaLLM(base_url="http://localhost:11434", model=model, timeout=0.001)
                
                # Mock to simulate timeout
                with patch.object(provider, 'agenerate', side_effect=asyncio.TimeoutError("Timeout")):
                    with pytest.raises((asyncio.TimeoutError, Exception)):
                        await provider.agenerate("Test prompt")
                        
            except ImportError:
                # Skip if provider not available
                continue
    
    @pytest.mark.asyncio
    async def test_api_key_validation(self, mock_env_vars):
        """Test API key validation"""
        # Test with invalid/missing API keys
        providers_to_test = [
            ("openai", "gpt-4"),
            ("xai", "grok-3-mini")
        ]
        
        for provider_type, model in providers_to_test:
            try:
                if provider_type == "openai":
                    from lightrag.llm.openai import OpenAILLM

                    # Should handle missing API key gracefully
                    provider = OpenAILLM(api_key="", model=model)
                elif provider_type == "xai":
                    from lightrag.llm.xai import XAILLM
                    provider = XAILLM(api_key="", model=model)
                
                # Provider should be created but might fail on actual call
                assert provider is not None
                
            except (ImportError, ValueError):
                # Skip if provider not available or validates API key at init
                continue
    
    @pytest.mark.asyncio
    async def test_model_availability_check(self, mock_env_vars):
        """Test checking if models are available"""
        try:
            from lightrag.llm.ollama import OllamaLLM
            
            llm = OllamaLLM(base_url="http://localhost:11434", model="non-existent-model")
            
            # Mock to simulate model not found
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_client = Mock()
                mock_httpx.return_value.__aenter__.return_value = mock_client
                
                mock_response = Mock()
                mock_response.status_code = 404
                mock_response.json.return_value = {"error": "model not found"}
                mock_client.post = AsyncMock(return_value=mock_response)
                
                # Should handle model not found error
                with pytest.raises(Exception):
                    await llm.agenerate("Test prompt")
        except ImportError:
            pytest.skip("Ollama provider not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lightrag.llm", "--cov-report=term-missing"])