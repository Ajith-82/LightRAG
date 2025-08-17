"""
Unit tests for lightrag.operate module.

Tests core operations that are actually available in the module.
"""

import asyncio
import json
from typing import List
from unittest.mock import AsyncMock, Mock

import pytest

from lightrag.operate import (
    extract_entities,
    normalize_extracted_info,
    chunking_by_token_size,
    clean_str,
    compute_mdhash_id,
    compute_args_hash
)


class TestTextChunking:
    """Test text chunking operations."""

    def test_chunking_by_token_size(self):
        """Test text chunking by token size."""
        text = "This is a test document with multiple sentences. " * 10
        
        chunks = chunking_by_token_size(
            text=text,
            chunk_token_size=50,
            chunk_overlap_token_size=10,
            tiktoken_model_name="gpt-4"
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # First chunk should not be empty
        assert len(chunks[0].strip()) > 0

    def test_chunking_short_text(self):
        """Test chunking text shorter than chunk size."""
        text = "This is a short text."
        
        chunks = chunking_by_token_size(
            text=text,
            chunk_token_size=100,
            chunk_overlap_token_size=20,
            tiktoken_model_name="gpt-4"
        )
        
        assert len(chunks) >= 1
        assert chunks[0] == text

    def test_chunking_empty_text(self):
        """Test chunking empty text."""
        chunks = chunking_by_token_size(
            text="",
            chunk_token_size=100,
            chunk_overlap_token_size=20,
            tiktoken_model_name="gpt-4"
        )
        
        # Should handle empty text gracefully
        assert isinstance(chunks, list)


class TestEntityExtraction:
    """Test entity extraction operations."""

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        text = "John Smith works at Google."
        
        # Mock LLM response
        mock_llm_response = '''
        {
            "entities": [
                {"name": "John Smith", "type": "person", "description": "Employee"},
                {"name": "Google", "type": "organization", "description": "Technology company"}
            ]
        }
        '''
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return mock_llm_response
        
        entities = await extract_entities(
            text=text,
            entity_types=["person", "organization"],
            llm_func=mock_llm,
            entity_extract_max_gleaning=1
        )
        
        assert len(entities) == 2
        assert any(e["name"] == "John Smith" for e in entities)
        assert any(e["type"] == "organization" for e in entities)

    @pytest.mark.asyncio
    async def test_extract_entities_empty_response(self):
        """Test entity extraction with empty LLM response."""
        text = "Some text without clear entities."
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return '{"entities": []}'
        
        entities = await extract_entities(
            text=text,
            entity_types=["person"],
            llm_func=mock_llm,
            entity_extract_max_gleaning=1
        )
        
        assert isinstance(entities, list)
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_malformed_response(self):
        """Test entity extraction with malformed LLM response."""
        text = "Test text."
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return "Not valid JSON"
        
        # Should handle malformed response gracefully
        entities = await extract_entities(
            text=text,
            entity_types=["person"],
            llm_func=mock_llm,
            entity_extract_max_gleaning=1
        )
        
        assert isinstance(entities, list)


class TestUtilityFunctions:
    """Test utility functions from operate module."""

    def test_clean_str(self):
        """Test string cleaning utility."""
        # Test with extra whitespace
        assert clean_str("  hello  world  ") == "hello world"
        
        # Test with newlines and tabs
        assert clean_str("hello\n\tworld") == "hello world"
        
        # Test with None
        assert clean_str(None) == ""
        
        # Test with empty string
        assert clean_str("") == ""

    def test_compute_mdhash_id(self):
        """Test MD5 hash computation."""
        content = "test content"
        hash1 = compute_mdhash_id(content)
        hash2 = compute_mdhash_id(content)
        
        # Should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length
        
        # Different content should produce different hash
        different_hash = compute_mdhash_id("different content")
        assert hash1 != different_hash

    def test_compute_args_hash(self):
        """Test argument hash computation."""
        # Same arguments should produce same hash
        hash1 = compute_args_hash(arg1="value1", arg2=42)
        hash2 = compute_args_hash(arg1="value1", arg2=42)
        assert hash1 == hash2
        
        # Different arguments should produce different hash
        hash3 = compute_args_hash(arg1="different", arg2=42)
        assert hash1 != hash3
        
        # Order independence
        hash4 = compute_args_hash(arg2=42, arg1="value1")
        assert hash1 == hash4

    def test_normalize_extracted_info(self):
        """Test normalizing extracted information."""
        # Test with valid extraction
        info = {
            "entities": [
                {"name": "  John Doe  ", "type": "person"},
                {"name": "Google", "type": "  organization  "}
            ]
        }
        
        normalized = normalize_extracted_info(info)
        
        # Check whitespace is trimmed
        assert normalized["entities"][0]["name"] == "John Doe"
        assert normalized["entities"][1]["type"] == "organization"
        
        # Test with empty extraction
        empty_info = {"entities": []}
        normalized = normalize_extracted_info(empty_info)
        assert normalized == empty_info
        
        # Test with None
        assert normalize_extracted_info(None) is None