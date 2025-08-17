"""
Comprehensive unit tests for core LightRAG functionality.

This module tests the main LightRAG class including initialization,
document processing, querying, and storage management.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from lightrag import LightRAG, QueryParam
from lightrag.lightrag import always_get_an_event_loop
from lightrag.utils import EmbeddingFunc


class TestLightRAGInitialization:
    """Test LightRAG initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic LightRAG initialization with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            assert rag.working_dir == Path(tmpdir)
            assert rag.chunk_token_size == 1200
            assert rag.chunk_overlap_token_size == 100
            assert rag.tiktoken_model_name == "gpt-4o-mini"

    def test_initialization_with_custom_params(self):
        """Test LightRAG initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llm_func = Mock()
            embed_func = Mock()
            
            rag = LightRAG(
                working_dir=tmpdir,
                llm_model_func=llm_func,
                embedding_func=embed_func,
                chunk_token_size=2000,
                chunk_overlap_token_size=200,
                entity_extract_max_gleaning=5,
                tiktoken_model_name="gpt-4",
                graph_storage="networkx",
                vector_storage="faiss",
                enable_llm_cache=False
            )
            
            assert rag.working_dir == Path(tmpdir)
            assert rag.chunk_token_size == 2000
            assert rag.chunk_overlap_token_size == 200
            assert rag.entity_extract_max_gleaning == 5
            assert rag.tiktoken_model_name == "gpt-4"
            assert rag.enable_llm_cache is False

    def test_working_dir_creation(self):
        """Test that working directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "new_dir"
            assert not working_dir.exists()
            
            rag = LightRAG(working_dir=str(working_dir))
            assert working_dir.exists()
            assert working_dir.is_dir()

    def test_invalid_chunk_sizes(self):
        """Test validation of chunk size parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Overlap size should not exceed chunk size
            with pytest.raises(ValueError, match="overlap.*cannot be larger than.*chunk"):
                LightRAG(
                    working_dir=tmpdir,
                    chunk_token_size=100,
                    chunk_overlap_token_size=200
                )


class TestLightRAGStorage:
    """Test storage initialization and management."""

    @pytest.mark.asyncio
    async def test_storage_initialization(self):
        """Test async storage initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock storage components
            rag.full_docs = AsyncMock()
            rag.text_chunks = AsyncMock()
            rag.entities_vdb = AsyncMock()
            rag.relationships_vdb = AsyncMock()
            rag.chunks_vdb = AsyncMock()
            rag.chunk_entity_relation_graph = AsyncMock()
            
            await rag.initialize_storages()
            
            # Verify all storages were initialized
            rag.full_docs.initialize.assert_called_once()
            rag.text_chunks.initialize.assert_called_once()
            rag.entities_vdb.initialize.assert_called_once()
            rag.relationships_vdb.initialize.assert_called_once()
            rag.chunks_vdb.initialize.assert_called_once()
            rag.chunk_entity_relation_graph.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_storage_finalization(self):
        """Test async storage finalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock storage components
            rag.full_docs = AsyncMock()
            rag.text_chunks = AsyncMock()
            rag.entities_vdb = AsyncMock()
            rag.relationships_vdb = AsyncMock()
            rag.chunks_vdb = AsyncMock()
            rag.chunk_entity_relation_graph = AsyncMock()
            
            await rag.finalize_storages()
            
            # Verify all storages were finalized
            rag.full_docs.finalize.assert_called_once()
            rag.text_chunks.finalize.assert_called_once()
            rag.entities_vdb.finalize.assert_called_once()
            rag.relationships_vdb.finalize.assert_called_once()
            rag.chunks_vdb.finalize.assert_called_once()
            rag.chunk_entity_relation_graph.finalize.assert_called_once()


class TestLightRAGDocumentProcessing:
    """Test document insertion and processing."""

    @pytest.mark.asyncio
    async def test_insert_text(self):
        """Test inserting text content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock the extract_and_store_from_docs method
            with patch.object(rag, 'extract_and_store_from_docs', new=AsyncMock()) as mock_extract:
                test_text = "This is a test document about artificial intelligence."
                await rag.insert(test_text)
                
                mock_extract.assert_called_once()
                docs = mock_extract.call_args[0][0]
                assert len(docs) == 1
                assert docs[0]["content"] == test_text

    @pytest.mark.asyncio
    async def test_insert_batch(self):
        """Test batch document insertion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock the extract_and_store_from_docs method
            with patch.object(rag, 'extract_and_store_from_docs', new=AsyncMock()) as mock_extract:
                test_texts = [
                    "Document 1 about machine learning.",
                    "Document 2 about deep learning.",
                    "Document 3 about neural networks."
                ]
                
                await rag.insert_batch(test_texts)
                
                mock_extract.assert_called_once()
                docs = mock_extract.call_args[0][0]
                assert len(docs) == 3
                for i, doc in enumerate(docs):
                    assert doc["content"] == test_texts[i]

    @pytest.mark.asyncio
    async def test_insert_file(self, tmp_path):
        """Test inserting content from a file."""
        # Create a temporary test file
        test_file = tmp_path / "test_document.txt"
        test_content = "This is content from a file about quantum computing."
        test_file.write_text(test_content)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock the extract_and_store_from_docs method
            with patch.object(rag, 'extract_and_store_from_docs', new=AsyncMock()) as mock_extract:
                await rag.insert(str(test_file))
                
                mock_extract.assert_called_once()
                docs = mock_extract.call_args[0][0]
                assert len(docs) == 1
                assert docs[0]["content"] == test_content
                assert "test_document.txt" in docs[0].get("file_path", "")


class TestLightRAGQuerying:
    """Test query functionality."""

    @pytest.mark.asyncio
    async def test_query_local_mode(self):
        """Test local query mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock the query implementation
            with patch('lightrag.lightrag.local_query', new=AsyncMock(return_value="Local result")) as mock_query:
                result = await rag.query("What is machine learning?", mode="local")
                
                assert result == "Local result"
                mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_global_mode(self):
        """Test global query mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock the query implementation
            with patch('lightrag.lightrag.global_query', new=AsyncMock(return_value="Global result")) as mock_query:
                result = await rag.query("What are the main themes?", mode="global")
                
                assert result == "Global result"
                mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_hybrid_mode(self):
        """Test hybrid query mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Mock the query implementation
            with patch('lightrag.lightrag.hybrid_query', new=AsyncMock(return_value="Hybrid result")) as mock_query:
                result = await rag.query("Explain the relationship between X and Y", mode="hybrid")
                
                assert result == "Hybrid result"
                mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_custom_params(self):
        """Test query with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Create custom query parameters
            param = QueryParam(
                mode="local",
                only_need_context=True,
                response_type="Multiple Paragraphs",
                top_k=10,
                max_token_for_local_context=2000
            )
            
            with patch('lightrag.lightrag.local_query', new=AsyncMock(return_value="Context only")) as mock_query:
                result = await rag.query("Test query", param=param)
                
                assert result == "Context only"
                mock_query.assert_called_once()
                
                # Verify parameters were passed correctly
                call_args = mock_query.call_args[1]
                assert call_args['param'].only_need_context is True
                assert call_args['param'].top_k == 10

    @pytest.mark.asyncio
    async def test_invalid_query_mode(self):
        """Test that invalid query mode raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            with pytest.raises(ValueError, match="mode.*not supported"):
                await rag.query("Test query", mode="invalid_mode")


class TestLightRAGUtilities:
    """Test utility methods and helper functions."""

    def test_always_get_event_loop(self):
        """Test event loop retrieval utility."""
        loop = always_get_an_event_loop()
        assert loop is not None
        assert isinstance(loop, asyncio.AbstractEventLoop)

    @pytest.mark.asyncio
    async def test_embedding_function_wrapper(self):
        """Test embedding function wrapper."""
        # Create a mock embedding function
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.array([[0.1, 0.2, 0.3] for _ in texts])
        
        embed_func = EmbeddingFunc(
            func=mock_embed,
            embedding_dim=3,
            max_token_size=8192
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(
                working_dir=tmpdir,
                embedding_func=embed_func
            )
            
            # Test that embedding function is properly wrapped
            assert rag.embedding_func is not None
            assert rag.embedding_func.embedding_dim == 3
            assert rag.embedding_func.max_token_size == 8192

    def test_load_prompts(self):
        """Test prompt loading mechanism."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            # Check that prompts are loaded
            assert hasattr(rag, 'entity_extraction_prompt')
            assert hasattr(rag, 'summarize_prompt')
            assert hasattr(rag, 'entiti_continue_extraction_prompt')
            assert hasattr(rag, 'process_client_message_prompt')


class TestLightRAGErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_insert_empty_text(self):
        """Test inserting empty text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            with patch.object(rag, 'extract_and_store_from_docs', new=AsyncMock()) as mock_extract:
                # Empty string should still be processed
                await rag.insert("")
                mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_nonexistent_file(self):
        """Test inserting from a non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            with pytest.raises(FileNotFoundError):
                await rag.insert("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_query_empty_string(self):
        """Test querying with empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(working_dir=tmpdir)
            
            with patch('lightrag.lightrag.local_query', new=AsyncMock(return_value="")) as mock_query:
                result = await rag.query("", mode="local")
                
                # Should still process empty query
                mock_query.assert_called_once()

    def test_invalid_working_directory(self):
        """Test initialization with invalid working directory."""
        # Try to use a file as working directory
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(NotADirectoryError):
                LightRAG(working_dir=tmpfile.name)


class TestLightRAGCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_llm_cache_enabled(self):
        """Test LLM caching when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(
                working_dir=tmpdir,
                enable_llm_cache=True
            )
            
            # Verify cache is enabled
            assert rag.enable_llm_cache is True
            
            # Check cache directory is created
            cache_dir = Path(tmpdir) / "llm_cache"
            # Cache directory is created on first use, not on initialization

    @pytest.mark.asyncio
    async def test_llm_cache_disabled(self):
        """Test LLM caching when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = LightRAG(
                working_dir=tmpdir,
                enable_llm_cache=False
            )
            
            # Verify cache is disabled
            assert rag.enable_llm_cache is False


class TestLightRAGIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete document insertion and query workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock functions
            async def mock_llm(prompt: str, **kwargs) -> str:
                return "Mock LLM response"
            
            async def mock_embed(texts: List[str]) -> np.ndarray:
                return np.random.rand(len(texts), 128)
            
            embed_func = EmbeddingFunc(
                func=mock_embed,
                embedding_dim=128,
                max_token_size=8192
            )
            
            rag = LightRAG(
                working_dir=tmpdir,
                llm_model_func=mock_llm,
                embedding_func=embed_func
            )
            
            # Mock storage initialization
            with patch.object(rag, 'initialize_storages', new=AsyncMock()):
                with patch.object(rag, 'extract_and_store_from_docs', new=AsyncMock()):
                    with patch('lightrag.lightrag.local_query', new=AsyncMock(return_value="Query result")):
                        # Initialize
                        await rag.initialize_storages()
                        
                        # Insert document
                        await rag.insert("Test document content")
                        
                        # Query
                        result = await rag.query("Test query", mode="local")
                        
                        assert result == "Query result"