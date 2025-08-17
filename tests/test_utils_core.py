"""
Comprehensive unit tests for lightrag.utils module.

This module tests utility functions including text processing, caching,
hash computations, file operations, and token management.
"""

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
import tiktoken

from lightrag.utils import (
    CacheData,
    EmbeddingFunc,
    clean_str,
    normalize_extracted_info,
    compute_mdhash_id,
    compute_args_hash,
    handle_cache,
    build_file_path,
    truncate_list_by_token_size,
    split_text_by_tokens,
    merge_text_chunks,
    wrap_text,
    sanitize_text,
    extract_metadata,
    format_timestamp,
    validate_json_response,
    retry_async,
    batch_process,
    normalize_whitespace,
    remove_duplicates,
    safe_json_parse,
    ensure_list,
    deep_merge_dict,
    flatten_dict,
    get_nested_value,
    set_nested_value
)


class TestTextProcessing:
    """Test text processing utilities."""

    def test_clean_str_basic(self):
        """Test basic string cleaning."""
        # Test with extra whitespace
        assert clean_str("  hello  world  ") == "hello world"
        
        # Test with newlines and tabs
        assert clean_str("hello\n\tworld") == "hello world"
        
        # Test with multiple spaces
        assert clean_str("hello     world") == "hello world"

    def test_clean_str_special_characters(self):
        """Test cleaning strings with special characters."""
        # Test with unicode characters
        assert clean_str("hello\u200bworld") == "helloworld"  # Zero-width space
        
        # Test with control characters
        assert clean_str("hello\x00world") == "helloworld"
        
        # Test preserving valid special characters
        assert clean_str("hello-world_123!") == "hello-world_123!"

    def test_clean_str_empty_and_none(self):
        """Test cleaning empty and None strings."""
        assert clean_str("") == ""
        assert clean_str(None) == ""
        assert clean_str("   ") == ""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "This  has\tmultiple\n\nspaces"
        normalized = normalize_whitespace(text)
        assert normalized == "This has multiple spaces"
        
        # Test preserving single spaces
        assert normalize_whitespace("already normalized") == "already normalized"

    def test_sanitize_text(self):
        """Test text sanitization."""
        # Test removing potentially harmful content
        text = "<script>alert('xss')</script>Hello world"
        sanitized = sanitize_text(text)
        assert "<script>" not in sanitized
        assert "Hello world" in sanitized
        
        # Test preserving safe content
        safe_text = "This is safe text with numbers 123"
        assert sanitize_text(safe_text) == safe_text

    def test_wrap_text(self):
        """Test text wrapping."""
        long_text = "This is a very long text that should be wrapped at a certain width to make it more readable."
        
        wrapped = wrap_text(long_text, width=20)
        lines = wrapped.split('\n')
        
        # Check that lines are wrapped
        assert len(lines) > 1
        assert all(len(line) <= 20 for line in lines)

    def test_remove_duplicates(self):
        """Test removing duplicate items from list."""
        items = ["apple", "banana", "apple", "orange", "banana", "grape"]
        unique = remove_duplicates(items)
        
        assert len(unique) == 4
        assert "apple" in unique
        assert "banana" in unique
        assert "orange" in unique
        assert "grape" in unique
        
        # Test preserving order
        assert unique.index("apple") < unique.index("banana")
        assert unique.index("banana") < unique.index("orange")


class TestHashingUtilities:
    """Test hashing and ID generation utilities."""

    def test_compute_mdhash_id(self):
        """Test MD5 hash ID computation."""
        # Test basic hashing
        content = "This is test content"
        hash_id = compute_mdhash_id(content)
        
        assert len(hash_id) == 32  # MD5 hash is 32 characters
        assert hash_id == compute_mdhash_id(content)  # Should be deterministic
        
        # Test different content produces different hash
        different_content = "Different content"
        different_hash = compute_mdhash_id(different_content)
        assert hash_id != different_hash

    def test_compute_mdhash_id_with_prefix(self):
        """Test MD5 hash with prefix."""
        content = "Test content"
        prefix = "doc"
        
        hash_id = compute_mdhash_id(content, prefix=prefix)
        assert hash_id.startswith(prefix)
        assert len(hash_id) > len(prefix)

    def test_compute_args_hash(self):
        """Test argument hash computation."""
        # Test with simple arguments
        args = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        hash1 = compute_args_hash(**args)
        
        # Same arguments should produce same hash
        hash2 = compute_args_hash(**args)
        assert hash1 == hash2
        
        # Different arguments should produce different hash
        different_args = {"key1": "value2", "key2": 42}
        hash3 = compute_args_hash(**different_args)
        assert hash1 != hash3

    def test_compute_args_hash_order_independence(self):
        """Test that argument order doesn't affect hash."""
        hash1 = compute_args_hash(a=1, b=2, c=3)
        hash2 = compute_args_hash(c=3, a=1, b=2)
        assert hash1 == hash2

    def test_compute_args_hash_complex_types(self):
        """Test hashing with complex argument types."""
        args = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (4, 5, 6),
            "none": None,
            "bool": True
        }
        
        hash_val = compute_args_hash(**args)
        assert len(hash_val) > 0
        
        # Modifying nested structure should change hash
        args["dict"]["nested"] = "different"
        different_hash = compute_args_hash(**args)
        assert hash_val != different_hash


class TestCaching:
    """Test caching utilities."""

    def test_cache_data_creation(self):
        """Test CacheData object creation."""
        data = {"key": "value", "number": 42}
        cache = CacheData(data=data, expires_in=3600)
        
        assert cache.data == data
        assert cache.expires_in == 3600
        assert cache.is_valid()

    def test_cache_data_expiration(self):
        """Test cache expiration."""
        cache = CacheData(data="test", expires_in=0)
        
        # Should be immediately expired
        import time
        time.sleep(0.1)
        assert not cache.is_valid()

    @pytest.mark.asyncio
    async def test_handle_cache_hit(self):
        """Test cache hit scenario."""
        cache_dir = Path(tempfile.mkdtemp())
        cache_key = "test_key"
        cached_data = {"result": "cached"}
        
        # Write cache file
        cache_file = cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps({
            "data": cached_data,
            "timestamp": asyncio.get_event_loop().time(),
            "expires_in": 3600
        }))
        
        # Test cache hit
        result = await handle_cache(
            cache_key=cache_key,
            cache_dir=cache_dir,
            func=AsyncMock(return_value={"result": "new"}),
            expires_in=3600
        )
        
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_handle_cache_miss(self):
        """Test cache miss scenario."""
        cache_dir = Path(tempfile.mkdtemp())
        cache_key = "missing_key"
        new_data = {"result": "new"}
        
        # Test cache miss
        result = await handle_cache(
            cache_key=cache_key,
            cache_dir=cache_dir,
            func=AsyncMock(return_value=new_data),
            expires_in=3600
        )
        
        assert result == new_data
        
        # Verify cache was written
        cache_file = cache_dir / f"{cache_key}.json"
        assert cache_file.exists()

    @pytest.mark.asyncio
    async def test_handle_cache_expired(self):
        """Test expired cache scenario."""
        cache_dir = Path(tempfile.mkdtemp())
        cache_key = "expired_key"
        old_data = {"result": "old"}
        new_data = {"result": "new"}
        
        # Write expired cache
        cache_file = cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps({
            "data": old_data,
            "timestamp": 0,  # Very old timestamp
            "expires_in": 1
        }))
        
        # Test with expired cache
        result = await handle_cache(
            cache_key=cache_key,
            cache_dir=cache_dir,
            func=AsyncMock(return_value=new_data),
            expires_in=3600
        )
        
        assert result == new_data


class TestFileOperations:
    """Test file operation utilities."""

    def test_build_file_path(self):
        """Test building file paths."""
        # Test basic path building
        path = build_file_path("dir", "subdir", "file.txt")
        assert str(path) == os.path.join("dir", "subdir", "file.txt")
        
        # Test with Path objects
        base = Path("base")
        path = build_file_path(base, "sub", "file.txt")
        assert path == base / "sub" / "file.txt"

    def test_build_file_path_absolute(self):
        """Test building absolute file paths."""
        abs_path = Path("/absolute/path")
        result = build_file_path(abs_path, "file.txt")
        assert result.is_absolute()
        assert str(result) == "/absolute/path/file.txt"

    def test_build_file_path_with_extension(self):
        """Test building paths with extension handling."""
        # Test adding extension
        path = build_file_path("dir", "file", ext=".txt")
        assert str(path).endswith("file.txt")
        
        # Test not duplicating extension
        path = build_file_path("dir", "file.txt", ext=".txt")
        assert str(path).endswith("file.txt")
        assert not str(path).endswith(".txt.txt")


class TestTokenManagement:
    """Test token-related utilities."""

    def test_truncate_list_by_token_size(self):
        """Test truncating list by token size."""
        items = ["short", "medium text", "this is a longer text", "very very long text indeed"]
        
        # Mock tokenizer
        def mock_token_count(text):
            return len(text.split())
        
        with patch('lightrag.utils.count_tokens', mock_token_count):
            truncated = truncate_list_by_token_size(
                items=items,
                max_tokens=10,
                tokenizer="mock"
            )
            
            # Should truncate to fit within token limit
            total_tokens = sum(len(item.split()) for item in truncated)
            assert total_tokens <= 10
            assert len(truncated) < len(items)

    def test_split_text_by_tokens(self):
        """Test splitting text by token count."""
        text = " ".join(["word"] * 100)  # 100 words
        
        def mock_token_count(text):
            return len(text.split())
        
        with patch('lightrag.utils.count_tokens', mock_token_count):
            chunks = split_text_by_tokens(
                text=text,
                max_tokens=20,
                overlap_tokens=5,
                tokenizer="mock"
            )
            
            assert len(chunks) > 1
            # Each chunk should be <= 20 tokens
            for chunk in chunks:
                assert len(chunk.split()) <= 20

    def test_merge_text_chunks(self):
        """Test merging text chunks."""
        chunks = [
            "This is chunk one.",
            "This is chunk two.",
            "This is chunk three."
        ]
        
        # Test basic merging
        merged = merge_text_chunks(chunks, separator=" ")
        assert merged == "This is chunk one. This is chunk two. This is chunk three."
        
        # Test with custom separator
        merged = merge_text_chunks(chunks, separator="\n")
        assert "\n" in merged
        
        # Test with empty chunks
        chunks_with_empty = ["First", "", "Third"]
        merged = merge_text_chunks(chunks_with_empty, separator=" ")
        assert merged == "First Third"


class TestJSONUtilities:
    """Test JSON-related utilities."""

    def test_safe_json_parse(self):
        """Test safe JSON parsing."""
        # Valid JSON
        valid_json = '{"key": "value", "number": 42}'
        result = safe_json_parse(valid_json)
        assert result == {"key": "value", "number": 42}
        
        # Invalid JSON should return None or default
        invalid_json = '{"key": invalid}'
        result = safe_json_parse(invalid_json, default={})
        assert result == {}
        
        # Non-string input
        result = safe_json_parse(None, default={})
        assert result == {}

    def test_validate_json_response(self):
        """Test JSON response validation."""
        # Valid response with required fields
        response = {"status": "success", "data": [1, 2, 3]}
        assert validate_json_response(
            response,
            required_fields=["status", "data"]
        ) is True
        
        # Missing required field
        response = {"status": "success"}
        assert validate_json_response(
            response,
            required_fields=["status", "data"]
        ) is False
        
        # Type validation
        response = {"count": "not_a_number"}
        assert validate_json_response(
            response,
            field_types={"count": int}
        ) is False

    def test_normalize_extracted_info(self):
        """Test normalizing extracted information."""
        # Test with valid extraction
        info = {
            "entities": [
                {"name": "  John Doe  ", "type": "person"},
                {"name": "Google", "type": "  organization  "}
            ],
            "relationships": [
                {"source": "John Doe", "target": "Google", "type": "WORKS_AT"}
            ]
        }
        
        normalized = normalize_extracted_info(info)
        
        # Check whitespace is trimmed
        assert normalized["entities"][0]["name"] == "John Doe"
        assert normalized["entities"][1]["type"] == "organization"
        
        # Test with empty extraction
        empty_info = {"entities": [], "relationships": []}
        normalized = normalize_extracted_info(empty_info)
        assert normalized == empty_info


class TestDictionaryUtilities:
    """Test dictionary manipulation utilities."""

    def test_deep_merge_dict(self):
        """Test deep merging of dictionaries."""
        dict1 = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "e": [1, 2]
        }
        
        dict2 = {
            "b": {"c": 4, "f": 5},
            "e": [3, 4],
            "g": 6
        }
        
        merged = deep_merge_dict(dict1, dict2)
        
        assert merged["a"] == 1  # From dict1
        assert merged["b"]["c"] == 4  # Overridden by dict2
        assert merged["b"]["d"] == 3  # From dict1
        assert merged["b"]["f"] == 5  # From dict2
        assert merged["g"] == 6  # From dict2
        assert len(merged["e"]) == 4  # Lists merged

    def test_flatten_dict(self):
        """Test flattening nested dictionary."""
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flattened = flatten_dict(nested)
        
        assert flattened["a"] == 1
        assert flattened["b.c"] == 2
        assert flattened["b.d.e"] == 3
        assert len(flattened) == 3

    def test_get_nested_value(self):
        """Test getting nested dictionary values."""
        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        
        # Test successful retrieval
        value = get_nested_value(data, "level1.level2.level3")
        assert value == "value"
        
        # Test with list of keys
        value = get_nested_value(data, ["level1", "level2", "level3"])
        assert value == "value"
        
        # Test missing key with default
        value = get_nested_value(data, "level1.missing", default="default")
        assert value == "default"

    def test_set_nested_value(self):
        """Test setting nested dictionary values."""
        data = {}
        
        # Set nested value
        set_nested_value(data, "level1.level2.level3", "value")
        assert data["level1"]["level2"]["level3"] == "value"
        
        # Overwrite existing value
        set_nested_value(data, "level1.level2.level3", "new_value")
        assert data["level1"]["level2"]["level3"] == "new_value"
        
        # Set with list of keys
        set_nested_value(data, ["level1", "new_key"], "another_value")
        assert data["level1"]["new_key"] == "another_value"


class TestListUtilities:
    """Test list manipulation utilities."""

    def test_ensure_list(self):
        """Test ensuring value is a list."""
        # Single value
        assert ensure_list("value") == ["value"]
        assert ensure_list(42) == [42]
        
        # Already a list
        assert ensure_list([1, 2, 3]) == [1, 2, 3]
        
        # None
        assert ensure_list(None) == []
        
        # Tuple
        assert ensure_list((1, 2, 3)) == [1, 2, 3]


class TestAsyncUtilities:
    """Test async utility functions."""

    @pytest.mark.asyncio
    async def test_retry_async(self):
        """Test async retry mechanism."""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_async(
            func=failing_func,
            max_retries=3,
            delay=0.1
        )
        
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_max_retries(self):
        """Test retry reaching max attempts."""
        async def always_failing():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            await retry_async(
                func=always_failing,
                max_retries=2,
                delay=0.1
            )

    @pytest.mark.asyncio
    async def test_batch_process(self):
        """Test batch processing of async tasks."""
        processed_items = []
        
        async def process_item(item):
            processed_items.append(item * 2)
            return item * 2
        
        items = [1, 2, 3, 4, 5]
        results = await batch_process(
            items=items,
            process_func=process_item,
            batch_size=2
        )
        
        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]
        assert len(processed_items) == 5


class TestEmbeddingFunc:
    """Test EmbeddingFunc wrapper."""

    @pytest.mark.asyncio
    async def test_embedding_func_wrapper(self):
        """Test EmbeddingFunc wrapper functionality."""
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 128)
        
        embed_func = EmbeddingFunc(
            func=mock_embed,
            embedding_dim=128,
            max_token_size=8192
        )
        
        # Test basic properties
        assert embed_func.embedding_dim == 128
        assert embed_func.max_token_size == 8192
        
        # Test calling the function
        texts = ["text1", "text2", "text3"]
        embeddings = await embed_func(texts)
        
        assert embeddings.shape == (3, 128)

    @pytest.mark.asyncio
    async def test_embedding_func_with_batching(self):
        """Test EmbeddingFunc with batching."""
        call_count = 0
        
        async def mock_embed(texts: List[str]) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.random.rand(len(texts), 64)
        
        embed_func = EmbeddingFunc(
            func=mock_embed,
            embedding_dim=64,
            max_token_size=100,
            batch_size=2
        )
        
        # Process more items than batch size
        texts = ["text1", "text2", "text3", "text4", "text5"]
        embeddings = await embed_func(texts)
        
        assert embeddings.shape == (5, 64)
        # Should be called multiple times due to batching
        assert call_count > 1


class TestMetadataExtraction:
    """Test metadata extraction utilities."""

    def test_extract_metadata(self):
        """Test extracting metadata from documents."""
        document = {
            "content": "This is the content",
            "source": "test.txt",
            "date": "2024-01-01",
            "author": "John Doe"
        }
        
        metadata = extract_metadata(
            document,
            fields=["source", "date", "author", "missing"]
        )
        
        assert metadata["source"] == "test.txt"
        assert metadata["date"] == "2024-01-01"
        assert metadata["author"] == "John Doe"
        assert "missing" not in metadata or metadata["missing"] is None

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        from datetime import datetime
        
        # Test with datetime object
        dt = datetime(2024, 1, 15, 10, 30, 45)
        formatted = format_timestamp(dt)
        assert "2024-01-15" in formatted
        assert "10:30:45" in formatted
        
        # Test with string timestamp
        timestamp_str = "2024-01-15T10:30:45Z"
        formatted = format_timestamp(timestamp_str)
        assert "2024-01-15" in formatted