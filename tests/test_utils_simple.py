"""
Unit tests for lightrag.utils module.

Tests utility functions that are actually available in the module.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from lightrag.utils import (
    CacheData,
    EmbeddingFunc,
    clean_str,
    build_file_path,
    always_get_an_event_loop
)


class TestStringUtilities:
    """Test string utility functions."""

    def test_clean_str_basic(self):
        """Test basic string cleaning."""
        # Test with leading/trailing whitespace
        assert clean_str("  hello world  ") == "hello world"
        
        # Test with newlines and tabs
        assert clean_str("hello\n\tworld") == "hello\tworld"
        
        # Test preserving internal spaces
        assert clean_str("  hello  world  ") == "hello  world"

    def test_clean_str_edge_cases(self):
        """Test edge cases in string cleaning."""
        # Test with empty string
        assert clean_str("") == ""
        
        # Test with None
        assert clean_str(None) == ""
        
        # Test with only whitespace
        assert clean_str("   ") == ""
        
        # Test with unicode whitespace
        assert clean_str("\u00A0hello\u00A0") == "hello"  # Non-breaking space


class TestFileUtilities:
    """Test file utility functions."""

    def test_build_file_path_basic(self):
        """Test basic file path building."""
        path = build_file_path("dir", "subdir", "file.txt")
        assert "dir" in str(path)
        assert "subdir" in str(path)
        assert "file.txt" in str(path)

    def test_build_file_path_with_pathlib(self):
        """Test file path building with Path objects."""
        base = Path("base")
        path = build_file_path(base, "sub", "file.txt")
        assert isinstance(path, Path)
        assert "base" in str(path)
        assert "sub" in str(path)
        assert "file.txt" in str(path)

    def test_build_file_path_absolute(self):
        """Test building absolute file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            abs_path = Path(tmpdir)
            result = build_file_path(abs_path, "file.txt")
            assert result.is_absolute()
            assert "file.txt" in str(result)


class TestCacheData:
    """Test CacheData class."""

    def test_cache_data_creation(self):
        """Test CacheData object creation."""
        data = {"key": "value", "number": 42}
        cache = CacheData(data=data)
        
        assert cache.data == data
        assert cache.is_valid() is True  # Should be valid by default

    def test_cache_data_with_expiration(self):
        """Test CacheData with expiration."""
        import time
        
        cache = CacheData(data="test", expires_in=0.1)
        assert cache.is_valid() is True
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.is_valid() is False

    def test_cache_data_no_expiration(self):
        """Test CacheData without expiration."""
        cache = CacheData(data="test")  # No expires_in
        assert cache.is_valid() is True
        
        # Should remain valid
        import time
        time.sleep(0.1)
        assert cache.is_valid() is True


class TestEmbeddingFunc:
    """Test EmbeddingFunc wrapper."""

    @pytest.mark.asyncio
    async def test_embedding_func_basic(self):
        """Test basic EmbeddingFunc functionality."""
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 128)
        
        embed_func = EmbeddingFunc(
            func=mock_embed,
            embedding_dim=128,
            max_token_size=8192
        )
        
        # Test properties
        assert embed_func.embedding_dim == 128
        assert embed_func.max_token_size == 8192
        
        # Test calling the function
        texts = ["text1", "text2", "text3"]
        embeddings = await embed_func(texts)
        
        assert embeddings.shape == (3, 128)
        assert isinstance(embeddings, np.ndarray)

    @pytest.mark.asyncio
    async def test_embedding_func_single_text(self):
        """Test EmbeddingFunc with single text."""
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 64)
        
        embed_func = EmbeddingFunc(
            func=mock_embed,
            embedding_dim=64
        )
        
        # Test with single text
        embedding = await embed_func(["single text"])
        assert embedding.shape == (1, 64)

    @pytest.mark.asyncio
    async def test_embedding_func_empty_texts(self):
        """Test EmbeddingFunc with empty text list."""
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 64)
        
        embed_func = EmbeddingFunc(
            func=mock_embed,
            embedding_dim=64
        )
        
        # Test with empty list
        embeddings = await embed_func([])
        assert embeddings.shape == (0, 64)

    def test_embedding_func_properties(self):
        """Test EmbeddingFunc properties."""
        def sync_embed(texts):
            return np.array([[0.1, 0.2]] * len(texts))
        
        embed_func = EmbeddingFunc(
            func=sync_embed,
            embedding_dim=2,
            max_token_size=1000
        )
        
        assert embed_func.embedding_dim == 2
        assert embed_func.max_token_size == 1000
        assert callable(embed_func.func)


class TestAsyncUtilities:
    """Test async utility functions."""

    def test_always_get_event_loop(self):
        """Test event loop retrieval utility."""
        loop = always_get_an_event_loop()
        assert loop is not None
        assert isinstance(loop, asyncio.AbstractEventLoop)

    def test_always_get_event_loop_multiple_calls(self):
        """Test multiple calls to get event loop."""
        loop1 = always_get_an_event_loop()
        loop2 = always_get_an_event_loop()
        
        # Should return the same loop
        assert loop1 is loop2

    @pytest.mark.asyncio
    async def test_event_loop_in_async_context(self):
        """Test getting event loop in async context."""
        loop = always_get_an_event_loop()
        current_loop = asyncio.get_running_loop()
        
        # Should return the running loop
        assert loop is current_loop


class TestIntegrationTests:
    """Integration tests combining multiple utilities."""

    @pytest.mark.asyncio
    async def test_embedding_with_file_paths(self):
        """Test embedding function with file path utilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock embedding function
            async def mock_embed(texts: List[str]) -> np.ndarray:
                return np.random.rand(len(texts), 100)
            
            embed_func = EmbeddingFunc(
                func=mock_embed,
                embedding_dim=100
            )
            
            # Build file path for storing embeddings
            embed_path = build_file_path(tmpdir, "embeddings", "test.npy")
            
            # Test embedding
            texts = ["sample text 1", "sample text 2"]
            embeddings = await embed_func(texts)
            
            # Verify embeddings
            assert embeddings.shape == (2, 100)
            
            # Verify path construction
            assert "embeddings" in str(embed_path)
            assert "test.npy" in str(embed_path)

    def test_clean_str_with_cache_data(self):
        """Test combining string cleaning with cache data."""
        # Create cache with dirty string
        dirty_data = {"text": "  hello  world  "}
        cache = CacheData(data=dirty_data)
        
        # Clean the cached string
        cleaned_text = clean_str(cache.data["text"])
        
        assert cleaned_text == "hello world"
        assert cache.is_valid() is True

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self):
        """Test a simulated workflow using multiple utilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Clean input text
            raw_text = "  This is a test document.  "
            cleaned_text = clean_str(raw_text)
            
            # Step 2: Create embedding function
            async def mock_embed(texts: List[str]) -> np.ndarray:
                # Simulate embedding based on text length
                return np.array([[len(text) / 100.0] * 50 for text in texts])
            
            embed_func = EmbeddingFunc(
                func=mock_embed,
                embedding_dim=50
            )
            
            # Step 3: Generate embeddings
            embeddings = await embed_func([cleaned_text])
            
            # Step 4: Cache the results
            cache = CacheData(
                data={
                    "text": cleaned_text,
                    "embeddings": embeddings.tolist()
                }
            )
            
            # Step 5: Build output path
            output_path = build_file_path(tmpdir, "results", "output.json")
            
            # Verify workflow
            assert cleaned_text == "This is a test document."
            assert embeddings.shape == (1, 50)
            assert cache.is_valid() is True
            assert "results" in str(output_path)
            assert "output.json" in str(output_path)