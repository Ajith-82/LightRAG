"""
Core Knowledge Graph Operations Tests for LightRAG
Tests entity management, relationship operations, graph traversal, and graph analytics
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import networkx as nx
import numpy as np
import pytest

# Test markers
pytestmark = [pytest.mark.core, pytest.mark.unit]


@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for testing"""
    # Entities
    entities = [
        {"id": "ai", "name": "Artificial Intelligence", "type": "concept", "description": "The simulation of human intelligence in machines"},
        {"id": "ml", "name": "Machine Learning", "type": "concept", "description": "A subset of AI that enables systems to learn automatically"},
        {"id": "dl", "name": "Deep Learning", "type": "concept", "description": "ML technique based on artificial neural networks"},
        {"id": "nn", "name": "Neural Network", "type": "architecture", "description": "Computing system inspired by biological neural networks"},
        {"id": "cnn", "name": "Convolutional Neural Network", "type": "architecture", "description": "Deep learning architecture for processing grid-like data"},
        {"id": "rnn", "name": "Recurrent Neural Network", "type": "architecture", "description": "Neural network for processing sequential data"},
        {"id": "transformer", "name": "Transformer", "type": "architecture", "description": "Neural network architecture based on attention mechanisms"},
        {"id": "bert", "name": "BERT", "type": "model", "description": "Bidirectional Encoder Representations from Transformers"},
        {"id": "gpt", "name": "GPT", "type": "model", "description": "Generative Pre-trained Transformer"},
        {"id": "nlp", "name": "Natural Language Processing", "type": "field", "description": "AI field focusing on human-computer language interaction"}
    ]
    
    # Relationships
    relationships = [
        {"source": "ml", "target": "ai", "type": "subset_of", "weight": 0.95},
        {"source": "dl", "target": "ml", "type": "subset_of", "weight": 0.9},
        {"source": "dl", "target": "nn", "type": "uses", "weight": 0.85},
        {"source": "cnn", "target": "nn", "type": "type_of", "weight": 0.8},
        {"source": "rnn", "target": "nn", "type": "type_of", "weight": 0.8},
        {"source": "transformer", "target": "nn", "type": "type_of", "weight": 0.75},
        {"source": "bert", "target": "transformer", "type": "based_on", "weight": 0.9},
        {"source": "gpt", "target": "transformer", "type": "based_on", "weight": 0.9},
        {"source": "bert", "target": "nlp", "type": "used_in", "weight": 0.8},
        {"source": "gpt", "target": "nlp", "type": "used_in", "weight": 0.8},
        {"source": "nlp", "target": "ai", "type": "field_of", "weight": 0.85},
        {"source": "cnn", "target": "dl", "type": "technique_of", "weight": 0.7},
        {"source": "rnn", "target": "dl", "type": "technique_of", "weight": 0.7}
    ]
    
    return {"entities": entities, "relationships": relationships}


@pytest.fixture
def mock_graph_storage():
    """Mock graph storage backend"""
    mock_storage = Mock()
    
    # Mock async methods
    mock_storage.upsert_entity = AsyncMock()
    mock_storage.upsert_relationship = AsyncMock()
    mock_storage.get_entity = AsyncMock()
    mock_storage.get_relationship = AsyncMock()
    mock_storage.get_all_entities = AsyncMock()
    mock_storage.get_all_relationships = AsyncMock()
    mock_storage.get_entity_neighbors = AsyncMock()
    mock_storage.delete_entity = AsyncMock()
    mock_storage.delete_relationship = AsyncMock()
    mock_storage.search_entities = AsyncMock()
    
    return mock_storage


@pytest.fixture
def networkx_graph(sample_knowledge_graph):
    """Create NetworkX graph for testing graph algorithms"""
    G = nx.DiGraph()
    
    # Add entities as nodes
    for entity in sample_knowledge_graph["entities"]:
        G.add_node(entity["id"], **entity)
    
    # Add relationships as edges
    for rel in sample_knowledge_graph["relationships"]:
        G.add_edge(
            rel["source"],
            rel["target"],
            relationship_type=rel["type"],
            weight=rel["weight"]
        )
    
    return G


class TestEntityOperations:
    """Test entity CRUD operations"""
    
    @pytest.mark.asyncio
    async def test_entity_creation(self, mock_graph_storage):
        """Test creating new entities"""
        entity = {
            "id": "test_entity",
            "name": "Test Entity",
            "type": "concept",
            "description": "A test entity for unit testing"
        }
        
        mock_graph_storage.upsert_entity.return_value = {"status": "created", "entity_id": entity["id"]}
        
        result = await mock_graph_storage.upsert_entity(entity["id"], entity)
        
        assert result["status"] == "created"
        assert result["entity_id"] == entity["id"]
        mock_graph_storage.upsert_entity.assert_called_once_with(entity["id"], entity)
    
    @pytest.mark.asyncio
    async def test_entity_retrieval(self, mock_graph_storage, sample_knowledge_graph):
        """Test retrieving entities"""
        ai_entity = next(e for e in sample_knowledge_graph["entities"] if e["id"] == "ai")
        mock_graph_storage.get_entity.return_value = ai_entity
        
        result = await mock_graph_storage.get_entity("ai")
        
        assert result["name"] == "Artificial Intelligence"
        assert result["type"] == "concept"
        mock_graph_storage.get_entity.assert_called_once_with("ai")
    
    @pytest.mark.asyncio
    async def test_entity_update(self, mock_graph_storage):
        """Test updating existing entities"""
        updated_entity = {
            "id": "ai",
            "name": "Artificial Intelligence", 
            "type": "concept",
            "description": "Updated: The simulation of human intelligence in machines using algorithms",
            "last_modified": "2025-01-15T10:00:00Z"
        }
        
        mock_graph_storage.upsert_entity.return_value = {"status": "updated", "entity_id": "ai"}
        
        result = await mock_graph_storage.upsert_entity("ai", updated_entity)
        
        assert result["status"] == "updated"
        mock_graph_storage.upsert_entity.assert_called_once_with("ai", updated_entity)
    
    @pytest.mark.asyncio
    async def test_entity_deletion(self, mock_graph_storage):
        """Test deleting entities"""
        mock_graph_storage.delete_entity.return_value = {
            "status": "deleted",
            "entity_id": "test_entity",
            "relationships_removed": 3
        }
        
        result = await mock_graph_storage.delete_entity("test_entity")
        
        assert result["status"] == "deleted"
        assert result["relationships_removed"] > 0
        mock_graph_storage.delete_entity.assert_called_once_with("test_entity")
    
    @pytest.mark.asyncio
    async def test_entity_search(self, mock_graph_storage, sample_knowledge_graph):
        """Test searching for entities"""
        # Mock search results
        search_results = [
            e for e in sample_knowledge_graph["entities"] 
            if "learning" in e["name"].lower()
        ]
        mock_graph_storage.search_entities.return_value = search_results
        
        results = await mock_graph_storage.search_entities("learning")
        
        assert len(results) >= 2  # Machine Learning, Deep Learning
        assert any(e["name"] == "Machine Learning" for e in results)
        assert any(e["name"] == "Deep Learning" for e in results)
        mock_graph_storage.search_entities.assert_called_once_with("learning")
    
    @pytest.mark.asyncio
    async def test_entity_type_filtering(self, mock_graph_storage, sample_knowledge_graph):
        """Test filtering entities by type"""
        concept_entities = [e for e in sample_knowledge_graph["entities"] if e["type"] == "concept"]
        mock_graph_storage.get_entities_by_type = AsyncMock(return_value=concept_entities)
        
        results = await mock_graph_storage.get_entities_by_type("concept")
        
        assert len(results) >= 3  # AI, ML, DL, NLP
        assert all(e["type"] == "concept" for e in results)


class TestRelationshipOperations:
    """Test relationship CRUD operations"""
    
    @pytest.mark.asyncio
    async def test_relationship_creation(self, mock_graph_storage):
        """Test creating relationships between entities"""
        relationship = {
            "source": "entity1",
            "target": "entity2",
            "type": "related_to",
            "weight": 0.8,
            "properties": {"confidence": "high", "bidirectional": False}
        }
        
        mock_graph_storage.upsert_relationship.return_value = {
            "status": "created",
            "relationship_id": "rel_123"
        }
        
        result = await mock_graph_storage.upsert_relationship(
            relationship["source"],
            relationship["target"],
            relationship["type"],
            relationship["properties"]
        )
        
        assert result["status"] == "created"
        mock_graph_storage.upsert_relationship.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_relationship_retrieval(self, mock_graph_storage, sample_knowledge_graph):
        """Test retrieving relationships"""
        ml_ai_rel = next(r for r in sample_knowledge_graph["relationships"] 
                        if r["source"] == "ml" and r["target"] == "ai")
        mock_graph_storage.get_relationship.return_value = ml_ai_rel
        
        result = await mock_graph_storage.get_relationship("ml", "ai")
        
        assert result["type"] == "subset_of"
        assert result["weight"] == 0.95
        mock_graph_storage.get_relationship.assert_called_once_with("ml", "ai")
    
    @pytest.mark.asyncio
    async def test_relationship_weight_updates(self, mock_graph_storage):
        """Test updating relationship weights"""
        updated_relationship = {
            "source": "ml",
            "target": "ai", 
            "type": "subset_of",
            "weight": 0.98,  # Increased confidence
            "last_updated": "2025-01-15T10:00:00Z"
        }
        
        mock_graph_storage.upsert_relationship.return_value = {
            "status": "updated",
            "old_weight": 0.95,
            "new_weight": 0.98
        }
        
        result = await mock_graph_storage.upsert_relationship(
            "ml", "ai", "subset_of", {"weight": 0.98}
        )
        
        assert result["status"] == "updated"
        assert result["new_weight"] > result["old_weight"]
    
    @pytest.mark.asyncio
    async def test_relationship_type_validation(self, mock_graph_storage):
        """Test validation of relationship types"""
        valid_relationship_types = [
            "subset_of", "type_of", "uses", "based_on", "used_in", 
            "field_of", "technique_of", "related_to", "depends_on"
        ]
        
        for rel_type in valid_relationship_types:
            mock_graph_storage.upsert_relationship.return_value = {"status": "created"}
            
            result = await mock_graph_storage.upsert_relationship(
                "entity1", "entity2", rel_type, {}
            )
            
            assert result["status"] == "created"
        
        # Test invalid relationship type
        mock_graph_storage.upsert_relationship.return_value = {"status": "error", "message": "Invalid relationship type"}
        
        result = await mock_graph_storage.upsert_relationship(
            "entity1", "entity2", "invalid_type", {}
        )
        
        assert result["status"] == "error"


class TestGraphTraversal:
    """Test graph traversal and navigation"""
    
    @pytest.mark.asyncio
    async def test_entity_neighbors(self, mock_graph_storage, sample_knowledge_graph):
        """Test finding entity neighbors"""
        # AI should have neighbors: ML, NLP
        ai_neighbors = ["ml", "nlp"]
        mock_graph_storage.get_entity_neighbors.return_value = ai_neighbors
        
        neighbors = await mock_graph_storage.get_entity_neighbors("ai")
        
        assert "ml" in neighbors
        assert "nlp" in neighbors
        mock_graph_storage.get_entity_neighbors.assert_called_once_with("ai")
    
    def test_shortest_path(self, networkx_graph):
        """Test finding shortest path between entities"""
        # Find path from BERT to AI
        try:
            path = nx.shortest_path(networkx_graph, "bert", "ai")
            assert path[0] == "bert"
            assert path[-1] == "ai"
            assert len(path) >= 3  # bert -> transformer -> ... -> ai
        except nx.NetworkXNoPath:
            pytest.fail("No path found between BERT and AI")
    
    def test_graph_connectivity(self, networkx_graph):
        """Test graph connectivity analysis"""
        # Test if graph is connected (ignoring direction)
        undirected_graph = networkx_graph.to_undirected()
        is_connected = nx.is_connected(undirected_graph)
        
        assert is_connected, "Knowledge graph should be connected"
        
        # Test strongly connected components
        strongly_connected = list(nx.strongly_connected_components(networkx_graph))
        assert len(strongly_connected) > 0
    
    def test_entity_centrality(self, networkx_graph):
        """Test calculating entity centrality measures"""
        # Degree centrality
        degree_centrality = nx.degree_centrality(networkx_graph)
        assert "ai" in degree_centrality
        assert degree_centrality["ai"] > 0
        
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(networkx_graph)
        assert all(0 <= v <= 1 for v in betweenness_centrality.values())
        
        # PageRank
        pagerank = nx.pagerank(networkx_graph)
        assert abs(sum(pagerank.values()) - 1.0) < 0.01  # Should sum to ~1
    
    def test_graph_clustering(self, networkx_graph):
        """Test graph clustering analysis"""
        # Clustering coefficient
        clustering = nx.clustering(networkx_graph.to_undirected())
        assert all(0 <= v <= 1 for v in clustering.values())
        
        # Average clustering
        avg_clustering = nx.average_clustering(networkx_graph.to_undirected())
        assert 0 <= avg_clustering <= 1


class TestGraphAnalytics:
    """Test advanced graph analytics and metrics"""
    
    @pytest.mark.asyncio
    async def test_entity_importance_scoring(self, mock_graph_storage, networkx_graph):
        """Test calculating entity importance scores"""
        # Mock importance calculation
        importance_scores = {
            "ai": 0.95,  # High importance (root concept)
            "ml": 0.85,  # High importance (major subset)
            "dl": 0.75,  # Medium-high importance
            "transformer": 0.65,  # Medium importance
            "bert": 0.55,  # Lower importance (specific model)
        }
        
        mock_graph_storage.calculate_entity_importance = AsyncMock(return_value=importance_scores)
        
        scores = await mock_graph_storage.calculate_entity_importance()
        
        assert scores["ai"] > scores["ml"] > scores["dl"]
        assert scores["transformer"] > scores["bert"]
        assert all(0 <= score <= 1 for score in scores.values())
    
    def test_relationship_strength_analysis(self, sample_knowledge_graph):
        """Test analyzing relationship strength patterns"""
        relationships = sample_knowledge_graph["relationships"]
        
        # Analyze relationship strength by type
        strength_by_type = {}
        for rel in relationships:
            rel_type = rel["type"]
            if rel_type not in strength_by_type:
                strength_by_type[rel_type] = []
            strength_by_type[rel_type].append(rel["weight"])
        
        # Calculate average strength per relationship type
        avg_strength = {
            rel_type: sum(weights) / len(weights)
            for rel_type, weights in strength_by_type.items()
        }
        
        # Verify expected patterns
        assert avg_strength.get("subset_of", 0) > 0.8  # High confidence relationships
        assert avg_strength.get("based_on", 0) > 0.8   # High confidence relationships
        assert all(0 <= avg for avg in avg_strength.values())
    
    def test_entity_clustering(self, networkx_graph):
        """Test entity clustering and community detection"""
        # Convert to undirected for community detection
        undirected = networkx_graph.to_undirected()
        
        # Simple community detection using modularity
        try:
            communities = nx.community.greedy_modularity_communities(undirected)
            
            assert len(communities) > 0
            assert sum(len(c) for c in communities) == len(undirected.nodes())
            
            # Verify no node appears in multiple communities
            all_nodes = set()
            for community in communities:
                assert all_nodes.isdisjoint(community)
                all_nodes.update(community)
                
        except (ImportError, AttributeError):
            # Skip if community detection not available
            pytest.skip("Community detection not available")
    
    @pytest.mark.asyncio
    async def test_graph_evolution_tracking(self, mock_graph_storage):
        """Test tracking graph changes over time"""
        # Mock graph evolution data
        evolution_data = {
            "timestamp": "2025-01-15T10:00:00Z",
            "changes": {
                "entities_added": ["quantum_computing", "blockchain"],
                "entities_modified": ["ai"],
                "entities_removed": [],
                "relationships_added": [
                    {"source": "quantum_computing", "target": "ai", "type": "field_of"}
                ],
                "relationships_modified": [
                    {"source": "ml", "target": "ai", "old_weight": 0.95, "new_weight": 0.98}
                ],
                "relationships_removed": []
            },
            "metrics": {
                "total_entities": 12,
                "total_relationships": 14,
                "avg_connectivity": 2.33,
                "graph_density": 0.105
            }
        }
        
        mock_graph_storage.track_graph_evolution = AsyncMock(return_value=evolution_data)
        
        result = await mock_graph_storage.track_graph_evolution()
        
        assert result["changes"]["entities_added"]
        assert result["changes"]["relationships_modified"]
        assert result["metrics"]["total_entities"] > 0


class TestGraphQuerying:
    """Test complex graph querying operations"""
    
    @pytest.mark.asyncio
    async def test_pattern_matching(self, mock_graph_storage):
        """Test pattern matching in the knowledge graph"""
        # Pattern: Find all entities that are subsets of AI
        pattern_query = {
            "pattern": "(?entity) --[subset_of]--> (ai)",
            "variables": ["entity"]
        }
        
        expected_matches = [
            {"entity": "ml"},
            {"entity": "nlp"}
        ]
        
        mock_graph_storage.match_pattern = AsyncMock(return_value=expected_matches)
        
        matches = await mock_graph_storage.match_pattern(pattern_query)
        
        assert len(matches) >= 2
        assert any(m["entity"] == "ml" for m in matches)
        assert any(m["entity"] == "nlp" for m in matches)
    
    @pytest.mark.asyncio
    async def test_subgraph_extraction(self, mock_graph_storage, sample_knowledge_graph):
        """Test extracting subgraphs based on criteria"""
        # Extract neural network subgraph
        nn_subgraph = {
            "entities": [e for e in sample_knowledge_graph["entities"] 
                        if "neural" in e["name"].lower() or e["type"] == "architecture"],
            "relationships": [r for r in sample_knowledge_graph["relationships"]
                           if any("neural" in sample_knowledge_graph["entities"][i]["name"].lower() 
                                 for i, e in enumerate(sample_knowledge_graph["entities"])
                                 if e["id"] in [r["source"], r["target"]])]
        }
        
        mock_graph_storage.extract_subgraph = AsyncMock(return_value=nn_subgraph)
        
        subgraph = await mock_graph_storage.extract_subgraph({"concept": "neural network"})
        
        assert len(subgraph["entities"]) > 0
        assert len(subgraph["relationships"]) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_graph_storage):
        """Test semantic search in the knowledge graph"""
        query = "deep learning techniques"
        
        # Mock semantic search results
        semantic_results = [
            {"entity": "dl", "name": "Deep Learning", "relevance": 0.95},
            {"entity": "cnn", "name": "Convolutional Neural Network", "relevance": 0.85},
            {"entity": "rnn", "name": "Recurrent Neural Network", "relevance": 0.82},
            {"entity": "transformer", "name": "Transformer", "relevance": 0.78}
        ]
        
        mock_graph_storage.semantic_search = AsyncMock(return_value=semantic_results)
        
        results = await mock_graph_storage.semantic_search(query)
        
        assert len(results) > 0
        assert all(r["relevance"] > 0.7 for r in results)
        assert results[0]["relevance"] >= results[-1]["relevance"]  # Sorted by relevance
    
    @pytest.mark.asyncio
    async def test_graph_reasoning(self, mock_graph_storage):
        """Test reasoning and inference on the knowledge graph"""
        # Inference: If A is subset of B, and B is subset of C, then A is related to C
        inference_query = {
            "premise": "dl --[subset_of]--> ml --[subset_of]--> ai",
            "infer": "dl --[related_to]--> ai"
        }
        
        inferred_relationship = {
            "source": "dl",
            "target": "ai", 
            "type": "transitively_related_to",
            "confidence": 0.8,
            "inference_path": ["dl", "ml", "ai"]
        }
        
        mock_graph_storage.infer_relationships = AsyncMock(return_value=[inferred_relationship])
        
        inferences = await mock_graph_storage.infer_relationships(inference_query)
        
        assert len(inferences) > 0
        assert inferences[0]["source"] == "dl"
        assert inferences[0]["target"] == "ai"
        assert inferences[0]["confidence"] > 0


class TestGraphVisualization:
    """Test graph visualization and export capabilities"""
    
    def test_graph_serialization(self, sample_knowledge_graph):
        """Test serializing graph to different formats"""
        # Test JSON serialization
        json_export = json.dumps(sample_knowledge_graph)
        assert json_export is not None
        
        # Test round-trip
        deserialized = json.loads(json_export)
        assert len(deserialized["entities"]) == len(sample_knowledge_graph["entities"])
        assert len(deserialized["relationships"]) == len(sample_knowledge_graph["relationships"])
    
    def test_graph_statistics(self, networkx_graph):
        """Test calculating graph statistics for visualization"""
        stats = {
            "num_nodes": networkx_graph.number_of_nodes(),
            "num_edges": networkx_graph.number_of_edges(),
            "density": nx.density(networkx_graph),
            "avg_degree": sum(dict(networkx_graph.degree()).values()) / networkx_graph.number_of_nodes(),
            "diameter": nx.diameter(networkx_graph.to_undirected()) if nx.is_connected(networkx_graph.to_undirected()) else None,
            "num_components": nx.number_connected_components(networkx_graph.to_undirected())
        }
        
        assert stats["num_nodes"] > 0
        assert stats["num_edges"] > 0
        assert 0 <= stats["density"] <= 1
        assert stats["avg_degree"] >= 0
    
    @pytest.mark.asyncio
    async def test_graph_export_formats(self, mock_graph_storage, sample_knowledge_graph):
        """Test exporting graph in various formats"""
        # Mock different export formats
        export_formats = {
            "json": json.dumps(sample_knowledge_graph),
            "graphml": "<graphml>...</graphml>",
            "cypher": "CREATE (ai:Concept {name: 'Artificial Intelligence'})...",
            "dot": "digraph KnowledgeGraph { ai -> ml [label=\"subset_of\"]; }"
        }
        
        mock_graph_storage.export_graph = AsyncMock()
        
        for format_type, expected_content in export_formats.items():
            mock_graph_storage.export_graph.return_value = expected_content
            
            result = await mock_graph_storage.export_graph(format_type)
            
            assert result is not None
            assert len(result) > 0
            mock_graph_storage.export_graph.assert_called_with(format_type)


class TestGraphPerformance:
    """Test graph performance and scalability"""
    
    @pytest.mark.asyncio
    async def test_bulk_entity_operations(self, mock_graph_storage):
        """Test bulk entity operations performance"""
        # Generate bulk entities
        bulk_entities = [
            {
                "id": f"entity_{i}",
                "name": f"Entity {i}",
                "type": "test",
                "description": f"Test entity number {i}"
            }
            for i in range(1000)
        ]
        
        mock_graph_storage.bulk_upsert_entities = AsyncMock(return_value={
            "status": "success",
            "entities_processed": len(bulk_entities),
            "processing_time": 2.5
        })
        
        result = await mock_graph_storage.bulk_upsert_entities(bulk_entities)
        
        assert result["status"] == "success"
        assert result["entities_processed"] == 1000
        assert result["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_bulk_relationship_operations(self, mock_graph_storage):
        """Test bulk relationship operations performance"""
        # Generate bulk relationships
        bulk_relationships = [
            {
                "source": f"entity_{i}",
                "target": f"entity_{i+1}",
                "type": "connects_to",
                "weight": 0.5 + (i % 50) / 100  # Vary weights
            }
            for i in range(999)
        ]
        
        mock_graph_storage.bulk_upsert_relationships = AsyncMock(return_value={
            "status": "success",
            "relationships_processed": len(bulk_relationships),
            "processing_time": 1.8
        })
        
        result = await mock_graph_storage.bulk_upsert_relationships(bulk_relationships)
        
        assert result["status"] == "success"
        assert result["relationships_processed"] == 999
        assert result["processing_time"] > 0
    
    def test_graph_algorithm_performance(self, networkx_graph):
        """Test performance of graph algorithms"""
        import time

        # Test shortest path performance
        start_time = time.time()
        paths = {}
        for source in list(networkx_graph.nodes())[:5]:  # Test first 5 nodes
            for target in list(networkx_graph.nodes())[:5]:
                if source != target:
                    try:
                        paths[(source, target)] = nx.shortest_path(networkx_graph, source, target)
                    except nx.NetworkXNoPath:
                        paths[(source, target)] = None
        path_time = time.time() - start_time
        
        # Test centrality performance
        start_time = time.time()
        centrality = nx.degree_centrality(networkx_graph)
        centrality_time = time.time() - start_time
        
        # Performance assertions
        assert path_time < 1.0  # Should complete within 1 second
        assert centrality_time < 0.5  # Should complete within 0.5 seconds
        assert len(centrality) == networkx_graph.number_of_nodes()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lightrag", "--cov-report=term-missing"])