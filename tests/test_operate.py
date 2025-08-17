"""
Comprehensive unit tests for lightrag.operate module.

This module tests core operations including entity extraction, relationship
extraction, text chunking, graph construction, and LLM integration patterns.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from lightrag.operate import (
    extract_entities,
    normalize_extracted_info,
    chunking_by_token_size,
    merge_nodes_and_edges,
    clean_str,
    compute_mdhash_id,
    compute_args_hash,
    handle_cache
)
from lightrag.utils import EmbeddingFunc


class TestTextChunking:
    """Test text chunking operations."""

    def test_basic_text_chunking(self):
        """Test basic text chunking with default parameters."""
        text = "This is a test document. " * 100  # Create long text
        
        chunks = chunking_by_token_size(
            text=text,
            chunk_token_size=100,
            chunk_overlap_token_size=20,
            tiktoken_model_name="gpt-4"
        )
        
        assert len(chunks) > 1
        # Check that we get chunks
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunking_with_no_overlap(self):
        """Test text chunking without overlap."""
        text = "Word " * 100
        
        chunks = chunk_text(
            text=text,
            chunk_size=50,
            overlap_size=0
        )
        
        assert len(chunks) > 1
        # No overlap means chunks should be distinct
        for i in range(len(chunks) - 1):
            assert chunks[i][-1] != chunks[i + 1][0]

    def test_chunking_short_text(self):
        """Test chunking text shorter than chunk size."""
        text = "This is a short text."
        
        chunks = chunk_text(
            text=text,
            chunk_size=100,
            overlap_size=20
        )
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunking_edge_cases(self):
        """Test edge cases in text chunking."""
        # Empty text
        chunks = chunk_text("", chunk_size=100, overlap_size=20)
        assert len(chunks) == 1
        assert chunks[0] == ""
        
        # Chunk size larger than text
        text = "Small text"
        chunks = chunk_text(text, chunk_size=1000, overlap_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_invalid_chunk_parameters(self):
        """Test validation of chunk parameters."""
        text = "Test text"
        
        # Overlap larger than chunk size should raise error
        with pytest.raises(ValueError, match="Overlap.*cannot be larger than chunk"):
            chunk_text(text, chunk_size=50, overlap_size=100)
        
        # Negative chunk size
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            chunk_text(text, chunk_size=-10, overlap_size=0)


class TestEntityExtraction:
    """Test entity extraction operations."""

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        text = "John Smith works at Google in Mountain View. He collaborates with Jane Doe on AI research."
        
        # Mock LLM response
        mock_llm_response = json.dumps({
            "entities": [
                {"name": "John Smith", "type": "person", "description": "Employee at Google"},
                {"name": "Google", "type": "organization", "description": "Technology company"},
                {"name": "Mountain View", "type": "location", "description": "City in California"},
                {"name": "Jane Doe", "type": "person", "description": "AI researcher"},
                {"name": "AI research", "type": "concept", "description": "Artificial intelligence research"}
            ]
        })
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return mock_llm_response
        
        entities = await extract_entities(
            text=text,
            llm_func=mock_llm,
            entity_types=["person", "organization", "location", "concept"]
        )
        
        assert len(entities) == 5
        assert any(e["name"] == "John Smith" for e in entities)
        assert any(e["type"] == "organization" for e in entities)

    @pytest.mark.asyncio
    async def test_extract_entities_with_gleaning(self):
        """Test entity extraction with multiple gleaning rounds."""
        text = "The conference on quantum computing was held in Tokyo."
        
        gleaning_responses = [
            json.dumps({
                "entities": [
                    {"name": "quantum computing", "type": "concept"},
                    {"name": "Tokyo", "type": "location"}
                ]
            }),
            json.dumps({
                "entities": [
                    {"name": "conference", "type": "event"}
                ]
            })
        ]
        
        call_count = 0
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            nonlocal call_count
            response = gleaning_responses[min(call_count, len(gleaning_responses) - 1)]
            call_count += 1
            return response
        
        entities = await extract_entities(
            text=text,
            llm_func=mock_llm,
            entity_types=["concept", "location", "event"],
            max_gleaning=2
        )
        
        # Should have entities from both gleaning rounds
        assert len(entities) == 3
        assert call_count == 2  # Should be called twice for gleaning

    @pytest.mark.asyncio
    async def test_extract_entities_error_handling(self):
        """Test entity extraction error handling."""
        text = "Test text"
        
        async def failing_llm(prompt: str, **kwargs) -> str:
            raise Exception("LLM API error")
        
        with pytest.raises(Exception, match="LLM API error"):
            await extract_entities(
                text=text,
                llm_func=failing_llm,
                entity_types=["person"]
            )

    @pytest.mark.asyncio
    async def test_entity_normalization(self):
        """Test entity normalization and deduplication."""
        entities = [
            {"name": "John Smith", "type": "person"},
            {"name": "john smith", "type": "person"},  # Duplicate with different case
            {"name": "J. Smith", "type": "person"},  # Potential duplicate
            {"name": "Google Inc.", "type": "organization"},
            {"name": "Google", "type": "organization"}  # Potential duplicate
        ]
        
        normalized = normalize_entities(entities, similarity_threshold=0.8)
        
        # Should merge similar entities
        assert len(normalized) < len(entities)
        
        # Check that normalization preserves entity types
        person_entities = [e for e in normalized if e["type"] == "person"]
        org_entities = [e for e in normalized if e["type"] == "organization"]
        assert len(person_entities) > 0
        assert len(org_entities) > 0


class TestRelationshipExtraction:
    """Test relationship extraction operations."""

    @pytest.mark.asyncio
    async def test_extract_relationships_basic(self):
        """Test basic relationship extraction."""
        text = "John Smith is the CEO of TechCorp. He founded the company in 2020."
        entities = [
            {"name": "John Smith", "type": "person"},
            {"name": "TechCorp", "type": "organization"}
        ]
        
        mock_llm_response = json.dumps({
            "relationships": [
                {
                    "source": "John Smith",
                    "target": "TechCorp",
                    "type": "CEO_OF",
                    "description": "John Smith is the CEO of TechCorp"
                },
                {
                    "source": "John Smith",
                    "target": "TechCorp",
                    "type": "FOUNDED",
                    "description": "John Smith founded TechCorp in 2020"
                }
            ]
        })
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return mock_llm_response
        
        relationships = await extract_relationships(
            text=text,
            entities=entities,
            llm_func=mock_llm
        )
        
        assert len(relationships) == 2
        assert any(r["type"] == "CEO_OF" for r in relationships)
        assert any(r["type"] == "FOUNDED" for r in relationships)

    @pytest.mark.asyncio
    async def test_extract_relationships_no_entities(self):
        """Test relationship extraction with no entities."""
        text = "This is a test text."
        entities = []
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return json.dumps({"relationships": []})
        
        relationships = await extract_relationships(
            text=text,
            entities=entities,
            llm_func=mock_llm
        )
        
        assert len(relationships) == 0

    @pytest.mark.asyncio
    async def test_relationship_validation(self):
        """Test relationship validation and cleaning."""
        relationships = [
            {
                "source": "John Smith",
                "target": "TechCorp",
                "type": "WORKS_AT",
                "description": "John works at TechCorp"
            },
            {
                "source": "Unknown Entity",  # Invalid - entity doesn't exist
                "target": "TechCorp",
                "type": "RELATED_TO",
                "description": "Some relationship"
            },
            {
                "source": "John Smith",
                "target": "John Smith",  # Self-relationship
                "type": "SAME_AS",
                "description": "Same person"
            }
        ]
        
        entities = [
            {"name": "John Smith", "type": "person"},
            {"name": "TechCorp", "type": "organization"}
        ]
        
        validated = clean_and_validate_relationships(relationships, entities)
        
        # Should remove invalid relationships
        assert len(validated) < len(relationships)
        # Should keep valid relationship
        assert any(r["source"] == "John Smith" and r["target"] == "TechCorp" for r in validated)


class TestKnowledgeGraphConstruction:
    """Test knowledge graph construction."""

    @pytest.mark.asyncio
    async def test_build_knowledge_graph(self):
        """Test building knowledge graph from entities and relationships."""
        entities = [
            {"name": "Alice", "type": "person", "description": "Software engineer"},
            {"name": "Bob", "type": "person", "description": "Data scientist"},
            {"name": "TechCorp", "type": "organization", "description": "Tech company"}
        ]
        
        relationships = [
            {
                "source": "Alice",
                "target": "TechCorp",
                "type": "WORKS_AT",
                "description": "Alice works at TechCorp"
            },
            {
                "source": "Bob",
                "target": "TechCorp",
                "type": "WORKS_AT",
                "description": "Bob works at TechCorp"
            },
            {
                "source": "Alice",
                "target": "Bob",
                "type": "COLLABORATES_WITH",
                "description": "Alice collaborates with Bob"
            }
        ]
        
        graph = await build_knowledge_graph(entities, relationships)
        
        # Check graph structure
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 3
        
        # Check node properties
        alice_node = next(n for n in graph["nodes"] if n["name"] == "Alice")
        assert alice_node["type"] == "person"
        assert alice_node["description"] == "Software engineer"
        
        # Check edge properties
        works_at_edges = [e for e in graph["edges"] if e["type"] == "WORKS_AT"]
        assert len(works_at_edges) == 2

    @pytest.mark.asyncio
    async def test_merge_graphs(self):
        """Test merging multiple knowledge graphs."""
        graph1 = {
            "nodes": [
                {"name": "Alice", "type": "person"},
                {"name": "TechCorp", "type": "organization"}
            ],
            "edges": [
                {"source": "Alice", "target": "TechCorp", "type": "WORKS_AT"}
            ]
        }
        
        graph2 = {
            "nodes": [
                {"name": "Bob", "type": "person"},
                {"name": "TechCorp", "type": "organization"},  # Duplicate
                {"name": "DataCorp", "type": "organization"}
            ],
            "edges": [
                {"source": "Bob", "target": "DataCorp", "type": "WORKS_AT"},
                {"source": "Bob", "target": "TechCorp", "type": "CONSULTS_FOR"}
            ]
        }
        
        merged = merge_graphs([graph1, graph2])
        
        # Should merge nodes and edges
        assert len(merged["nodes"]) == 4  # Alice, Bob, TechCorp (deduped), DataCorp
        assert len(merged["edges"]) == 3
        
        # Check that TechCorp is not duplicated
        techcorp_nodes = [n for n in merged["nodes"] if n["name"] == "TechCorp"]
        assert len(techcorp_nodes) == 1

    def test_optimize_graph_structure(self):
        """Test graph structure optimization."""
        graph = {
            "nodes": [
                {"name": "A", "type": "concept", "weight": 0.1},
                {"name": "B", "type": "concept", "weight": 0.9},
                {"name": "C", "type": "concept", "weight": 0.5},
                {"name": "D", "type": "concept", "weight": 0.05}  # Low weight, should be pruned
            ],
            "edges": [
                {"source": "A", "target": "B", "weight": 0.8},
                {"source": "B", "target": "C", "weight": 0.7},
                {"source": "C", "target": "D", "weight": 0.1},  # Low weight edge
                {"source": "A", "target": "D", "weight": 0.05}  # Very low weight
            ]
        }
        
        optimized = optimize_graph_structure(
            graph,
            min_node_weight=0.1,
            min_edge_weight=0.2
        )
        
        # Should remove low-weight nodes and edges
        assert len(optimized["nodes"]) < len(graph["nodes"])
        assert len(optimized["edges"]) < len(graph["edges"])
        
        # Check that high-weight nodes are retained
        assert any(n["name"] == "B" for n in optimized["nodes"])
        # Check that low-weight node is removed
        assert not any(n["name"] == "D" for n in optimized["nodes"])


class TestDocumentProcessing:
    """Test document processing operations."""

    @pytest.mark.asyncio
    async def test_process_documents(self):
        """Test processing multiple documents."""
        documents = [
            {"content": "Document 1 about AI.", "metadata": {"source": "doc1.txt"}},
            {"content": "Document 2 about ML.", "metadata": {"source": "doc2.txt"}},
            {"content": "Document 3 about DL.", "metadata": {"source": "doc3.txt"}}
        ]
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return json.dumps({
                "entities": [{"name": "AI", "type": "concept"}],
                "relationships": []
            })
        
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 128)
        
        results = await process_documents(
            documents=documents,
            llm_func=mock_llm,
            embedding_func=mock_embed,
            chunk_size=100,
            overlap_size=20
        )
        
        assert len(results) == 3
        assert all("entities" in r for r in results)
        assert all("relationships" in r for r in results)
        assert all("chunks" in r for r in results)

    @pytest.mark.asyncio
    async def test_extract_and_store(self):
        """Test extract and store operation."""
        document = {
            "content": "This is a test document about quantum computing.",
            "metadata": {"source": "test.txt", "date": "2024-01-01"}
        }
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return json.dumps({
                "entities": [
                    {"name": "quantum computing", "type": "concept"}
                ],
                "relationships": []
            })
        
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.random.rand(len(texts), 128)
        
        # Mock storage
        mock_storage = AsyncMock()
        
        await extract_and_store(
            document=document,
            llm_func=mock_llm,
            embedding_func=mock_embed,
            storage=mock_storage,
            chunk_size=100
        )
        
        # Verify storage was called
        mock_storage.store.assert_called()

    @pytest.mark.asyncio
    async def test_process_empty_document(self):
        """Test processing empty document."""
        document = {"content": "", "metadata": {}}
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return json.dumps({"entities": [], "relationships": []})
        
        async def mock_embed(texts: List[str]) -> np.ndarray:
            return np.array([[0.0] * 128 for _ in texts])
        
        result = await process_documents(
            documents=[document],
            llm_func=mock_llm,
            embedding_func=mock_embed,
            chunk_size=100
        )
        
        assert len(result) == 1
        assert len(result[0]["entities"]) == 0
        assert len(result[0]["relationships"]) == 0


class TestEntitySimilarity:
    """Test entity similarity computation."""

    def test_compute_entity_similarity(self):
        """Test computing similarity between entities."""
        entity1 = {
            "name": "John Smith",
            "type": "person",
            "embedding": np.array([0.1, 0.2, 0.3])
        }
        
        entity2 = {
            "name": "J. Smith",
            "type": "person",
            "embedding": np.array([0.15, 0.25, 0.35])
        }
        
        entity3 = {
            "name": "Jane Doe",
            "type": "person",
            "embedding": np.array([0.8, 0.9, 0.7])
        }
        
        # Similar entities should have high similarity
        sim_12 = compute_entity_similarity(entity1, entity2)
        assert sim_12 > 0.9
        
        # Different entities should have lower similarity
        sim_13 = compute_entity_similarity(entity1, entity3)
        assert sim_13 < sim_12

    def test_entity_similarity_different_types(self):
        """Test similarity computation for entities of different types."""
        person_entity = {
            "name": "Smith",
            "type": "person",
            "embedding": np.array([0.1, 0.2, 0.3])
        }
        
        org_entity = {
            "name": "Smith",
            "type": "organization",
            "embedding": np.array([0.1, 0.2, 0.3])
        }
        
        # Even with same name and embedding, different types should reduce similarity
        similarity = compute_entity_similarity(
            person_entity,
            org_entity,
            type_weight=0.5
        )
        
        assert similarity < 1.0  # Not perfect match due to type difference


class TestEntitySummary:
    """Test entity summary generation."""

    @pytest.mark.asyncio
    async def test_generate_entity_summary(self):
        """Test generating summary for an entity."""
        entity = {
            "name": "Quantum Computing",
            "type": "concept",
            "description": "A type of computation using quantum phenomena",
            "mentions": [
                "Quantum computing uses qubits",
                "It can solve certain problems exponentially faster",
                "Major companies are investing in quantum research"
            ]
        }
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return "Quantum computing is a revolutionary computational paradigm that leverages quantum mechanics."
        
        summary = await generate_entity_summary(entity, llm_func=mock_llm)
        
        assert len(summary) > 0
        assert "quantum" in summary.lower()

    @pytest.mark.asyncio
    async def test_generate_summary_minimal_entity(self):
        """Test generating summary for entity with minimal information."""
        entity = {
            "name": "Unknown Entity",
            "type": "unknown"
        }
        
        async def mock_llm(prompt: str, **kwargs) -> str:
            return "An entity with limited information available."
        
        summary = await generate_entity_summary(entity, llm_func=mock_llm)
        
        assert len(summary) > 0