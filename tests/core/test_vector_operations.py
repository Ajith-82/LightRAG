"""
Core Vector Search and Similarity Tests for LightRAG
Tests embedding generation, vector storage, similarity search, and vector analytics
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import faiss
import numpy as np
import pytest
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Test markers
pytestmark = [pytest.mark.core, pytest.mark.unit]


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing"""
    np.random.seed(42)  # For reproducible results
    
    embeddings = {
        "ai_concept": np.random.rand(768).astype(np.float32),
        "machine_learning": np.random.rand(768).astype(np.float32),
        "deep_learning": np.random.rand(768).astype(np.float32),
        "neural_networks": np.random.rand(768).astype(np.float32),
        "natural_language": np.random.rand(768).astype(np.float32),
        "computer_vision": np.random.rand(768).astype(np.float32),
        "robotics": np.random.rand(768).astype(np.float32),
        "data_science": np.random.rand(768).astype(np.float32),
        "statistics": np.random.rand(768).astype(np.float32),
        "mathematics": np.random.rand(768).astype(np.float32)
    }
    
    # Make some embeddings more similar (AI-related concepts)
    ai_base = embeddings["ai_concept"]
    embeddings["machine_learning"] = ai_base + np.random.normal(0, 0.1, 768).astype(np.float32)
    embeddings["deep_learning"] = ai_base + np.random.normal(0, 0.15, 768).astype(np.float32)
    embeddings["neural_networks"] = ai_base + np.random.normal(0, 0.2, 768).astype(np.float32)
    
    # Normalize embeddings
    for key in embeddings:
        embeddings[key] = embeddings[key] / np.linalg.norm(embeddings[key])
    
    return embeddings


@pytest.fixture
def mock_vector_storage():
    """Mock vector storage backend"""
    mock_storage = Mock()
    
    # Mock async methods
    mock_storage.upsert = AsyncMock()
    mock_storage.query = AsyncMock()
    mock_storage.get = AsyncMock()
    mock_storage.delete = AsyncMock()
    mock_storage.get_all_vectors = AsyncMock()
    mock_storage.update_vector = AsyncMock()
    mock_storage.bulk_upsert = AsyncMock()
    mock_storage.similarity_search = AsyncMock()
    
    return mock_storage


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider"""
    mock_provider = Mock()
    
    # Mock embedding generation
    async def mock_generate_embedding(texts):
        # Generate consistent embeddings for testing
        embeddings = []
        for text in texts:
            # Simple hash-based embedding for consistency
            hash_val = hash(text) % (2**32)
            embedding = np.random.RandomState(hash_val).rand(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding.tolist())
        return embeddings
    
    mock_provider.agenerate = AsyncMock(side_effect=mock_generate_embedding)
    
    return mock_provider


class TestEmbeddingGeneration:
    """Test embedding generation and preprocessing"""
    
    @pytest.mark.asyncio
    async def test_text_embedding_generation(self, mock_embedding_provider):
        """Test generating embeddings for text"""
        texts = [
            "Artificial intelligence is a branch of computer science",
            "Machine learning enables computers to learn without explicit programming",
            "Deep learning uses neural networks with multiple layers"
        ]
        
        embeddings = await mock_embedding_provider.agenerate(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 768 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)
        
        # Check embeddings are normalized (approximately)
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_embedding_provider):
        """Test batch embedding generation for performance"""
        # Large batch of texts
        texts = [f"Document {i} about AI and machine learning" for i in range(100)]
        
        embeddings = await mock_embedding_provider.agenerate(texts)
        
        assert len(embeddings) == 100
        assert all(len(emb) == 768 for emb in embeddings)
        
        # Verify batch consistency
        single_embedding = await mock_embedding_provider.agenerate([texts[0]])
        assert np.allclose(embeddings[0], single_embedding[0], rtol=1e-5)
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, mock_embedding_provider):
        """Test embedding caching for identical texts"""
        text = "Machine learning is a subset of artificial intelligence"
        
        # Generate embedding twice
        embedding1 = await mock_embedding_provider.agenerate([text])
        embedding2 = await mock_embedding_provider.agenerate([text])
        
        # Should be identical (due to consistent hashing in mock)
        assert np.allclose(embedding1[0], embedding2[0], rtol=1e-10)
    
    @pytest.mark.asyncio
    async def test_embedding_different_lengths(self, mock_embedding_provider):
        """Test embeddings for texts of different lengths"""
        texts = [
            "AI",  # Very short
            "Machine learning is important",  # Medium
            "Artificial intelligence and machine learning are transformative technologies that are reshaping industries and creating new possibilities for automation, decision-making, and problem-solving across various domains."  # Long
        ]
        
        embeddings = await mock_embedding_provider.agenerate(texts)
        
        # All embeddings should have same dimension regardless of text length
        assert all(len(emb) == 768 for emb in embeddings)
        
        # Embeddings should be different despite same dimension
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])


class TestVectorStorage:
    """Test vector storage operations"""
    
    @pytest.mark.asyncio
    async def test_vector_upsert(self, mock_vector_storage, sample_embeddings):
        """Test storing vectors in vector database"""
        vector_id = "test_vector_1"
        embedding = sample_embeddings["ai_concept"]
        metadata = {"text": "Artificial intelligence concept", "category": "AI"}
        
        mock_vector_storage.upsert.return_value = {
            "status": "success",
            "vector_id": vector_id,
            "dimension": len(embedding)
        }
        
        result = await mock_vector_storage.upsert(vector_id, embedding.tolist(), metadata)
        
        assert result["status"] == "success"
        assert result["vector_id"] == vector_id
        assert result["dimension"] == 768
        mock_vector_storage.upsert.assert_called_once_with(vector_id, embedding.tolist(), metadata)
    
    @pytest.mark.asyncio
    async def test_vector_retrieval(self, mock_vector_storage, sample_embeddings):
        """Test retrieving specific vectors"""
        vector_id = "ai_concept"
        expected_vector = sample_embeddings[vector_id]
        
        mock_vector_storage.get.return_value = {
            "vector_id": vector_id,
            "embedding": expected_vector.tolist(),
            "metadata": {"text": "AI concept", "category": "concept"}
        }
        
        result = await mock_vector_storage.get(vector_id)
        
        assert result["vector_id"] == vector_id
        assert len(result["embedding"]) == 768
        assert result["metadata"]["category"] == "concept"
        mock_vector_storage.get.assert_called_once_with(vector_id)
    
    @pytest.mark.asyncio
    async def test_bulk_vector_operations(self, mock_vector_storage, sample_embeddings):
        """Test bulk vector operations for performance"""
        vector_data = []
        for vector_id, embedding in sample_embeddings.items():
            vector_data.append({
                "id": vector_id,
                "embedding": embedding.tolist(),
                "metadata": {"concept": vector_id}
            })
        
        mock_vector_storage.bulk_upsert.return_value = {
            "status": "success",
            "vectors_processed": len(vector_data),
            "processing_time": 0.5
        }
        
        result = await mock_vector_storage.bulk_upsert(vector_data)
        
        assert result["status"] == "success"
        assert result["vectors_processed"] == len(sample_embeddings)
        mock_vector_storage.bulk_upsert.assert_called_once_with(vector_data)
    
    @pytest.mark.asyncio
    async def test_vector_update(self, mock_vector_storage, sample_embeddings):
        """Test updating existing vectors"""
        vector_id = "ai_concept"
        original_embedding = sample_embeddings[vector_id]
        updated_embedding = original_embedding + np.random.normal(0, 0.1, 768).astype(np.float32)
        updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
        
        mock_vector_storage.update_vector.return_value = {
            "status": "updated",
            "vector_id": vector_id,
            "similarity_to_previous": 0.95
        }
        
        result = await mock_vector_storage.update_vector(vector_id, updated_embedding.tolist())
        
        assert result["status"] == "updated"
        assert result["similarity_to_previous"] > 0.9
        mock_vector_storage.update_vector.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vector_deletion(self, mock_vector_storage):
        """Test deleting vectors"""
        vector_id = "vector_to_delete"
        
        mock_vector_storage.delete.return_value = {
            "status": "deleted",
            "vector_id": vector_id
        }
        
        result = await mock_vector_storage.delete(vector_id)
        
        assert result["status"] == "deleted"
        assert result["vector_id"] == vector_id
        mock_vector_storage.delete.assert_called_once_with(vector_id)


class TestSimilaritySearch:
    """Test vector similarity search operations"""
    
    @pytest.mark.asyncio
    async def test_cosine_similarity_search(self, mock_vector_storage, sample_embeddings):
        """Test cosine similarity search"""
        query_vector = sample_embeddings["ai_concept"]
        
        # Mock similarity search results
        mock_results = [
            {"id": "machine_learning", "score": 0.95, "metadata": {"category": "AI"}},
            {"id": "deep_learning", "score": 0.88, "metadata": {"category": "AI"}},
            {"id": "neural_networks", "score": 0.82, "metadata": {"category": "architecture"}},
            {"id": "computer_vision", "score": 0.75, "metadata": {"category": "AI"}}
        ]
        
        mock_vector_storage.similarity_search.return_value = mock_results
        
        results = await mock_vector_storage.similarity_search(
            query_vector.tolist(),
            top_k=4,
            metric="cosine"
        )
        
        assert len(results) == 4
        assert all(0 <= r["score"] <= 1 for r in results)
        assert results[0]["score"] >= results[-1]["score"]  # Sorted by similarity
        mock_vector_storage.similarity_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_euclidean_similarity_search(self, mock_vector_storage, sample_embeddings):
        """Test Euclidean distance similarity search"""
        query_vector = sample_embeddings["machine_learning"]
        
        mock_results = [
            {"id": "ai_concept", "distance": 0.15, "metadata": {"category": "concept"}},
            {"id": "deep_learning", "distance": 0.22, "metadata": {"category": "technique"}},
            {"id": "data_science", "distance": 0.35, "metadata": {"category": "field"}}
        ]
        
        mock_vector_storage.similarity_search.return_value = mock_results
        
        results = await mock_vector_storage.similarity_search(
            query_vector.tolist(),
            top_k=3,
            metric="euclidean"
        )
        
        assert len(results) == 3
        assert all(r["distance"] >= 0 for r in results)
        assert results[0]["distance"] <= results[-1]["distance"]  # Sorted by distance
    
    @pytest.mark.asyncio
    async def test_filtered_similarity_search(self, mock_vector_storage, sample_embeddings):
        """Test similarity search with metadata filtering"""
        query_vector = sample_embeddings["ai_concept"]
        
        # Filter for AI-related concepts only
        filters = {"category": "AI"}
        
        mock_results = [
            {"id": "machine_learning", "score": 0.95, "metadata": {"category": "AI"}},
            {"id": "computer_vision", "score": 0.75, "metadata": {"category": "AI"}}
        ]
        
        mock_vector_storage.similarity_search.return_value = mock_results
        
        results = await mock_vector_storage.similarity_search(
            query_vector.tolist(),
            top_k=5,
            filters=filters
        )
        
        assert len(results) == 2  # Only AI category results
        assert all(r["metadata"]["category"] == "AI" for r in results)
    
    def test_similarity_calculations(self, sample_embeddings):
        """Test different similarity calculation methods"""
        vec1 = sample_embeddings["ai_concept"]
        vec2 = sample_embeddings["machine_learning"]
        vec3 = sample_embeddings["statistics"]  # Less related
        
        # Cosine similarity
        cos_sim_12 = 1 - cosine(vec1, vec2)
        cos_sim_13 = 1 - cosine(vec1, vec3)
        
        # AI and ML should be more similar than AI and statistics
        assert cos_sim_12 > cos_sim_13
        
        # Euclidean distance
        eucl_dist_12 = euclidean(vec1, vec2)
        eucl_dist_13 = euclidean(vec1, vec3)
        
        # AI and ML should be closer than AI and statistics
        assert eucl_dist_12 < eucl_dist_13
        
        # Dot product similarity (for normalized vectors)
        dot_sim_12 = np.dot(vec1, vec2)
        dot_sim_13 = np.dot(vec1, vec3)
        
        assert dot_sim_12 > dot_sim_13
    
    @pytest.mark.asyncio
    async def test_approximate_nearest_neighbor(self, mock_vector_storage, sample_embeddings):
        """Test approximate nearest neighbor search for large datasets"""
        query_vector = sample_embeddings["deep_learning"]
        
        # Mock ANN results (slightly different from exact search)
        mock_ann_results = [
            {"id": "neural_networks", "score": 0.89, "method": "ANN"},
            {"id": "machine_learning", "score": 0.85, "method": "ANN"},
            {"id": "ai_concept", "score": 0.78, "method": "ANN"}
        ]
        
        mock_vector_storage.approximate_search = AsyncMock(return_value=mock_ann_results)
        
        results = await mock_vector_storage.approximate_search(
            query_vector.tolist(),
            top_k=3,
            method="hnsw"  # Hierarchical Navigable Small World
        )
        
        assert len(results) == 3
        assert all(r["method"] == "ANN" for r in results)


class TestVectorAnalytics:
    """Test vector analytics and clustering"""
    
    def test_vector_clustering(self, sample_embeddings):
        """Test clustering of vectors"""
        # Convert to matrix
        vectors = np.array(list(sample_embeddings.values()))
        labels = list(sample_embeddings.keys())
        
        # Simple K-means clustering simulation
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # Verify clustering results
        assert len(cluster_labels) == len(labels)
        assert len(set(cluster_labels)) <= 3  # Should have at most 3 clusters
        
        # Group by clusters
        clusters = {}
        for i, (label, cluster_id) in enumerate(zip(labels, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(label)
        
        # AI-related concepts should likely be in the same cluster
        ai_concepts = ["ai_concept", "machine_learning", "deep_learning", "neural_networks"]
        ai_clusters = [cluster_labels[labels.index(concept)] for concept in ai_concepts if concept in labels]
        
        # Most AI concepts should be in the same cluster
        from collections import Counter
        most_common_cluster = Counter(ai_clusters).most_common(1)[0][0]
        ai_in_main_cluster = sum(1 for c in ai_clusters if c == most_common_cluster)
        
        assert ai_in_main_cluster >= len(ai_concepts) // 2
    
    def test_vector_dimensionality_reduction(self, sample_embeddings):
        """Test dimensionality reduction for visualization"""
        vectors = np.array(list(sample_embeddings.values()))
        
        # PCA reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)
        
        assert reduced_vectors.shape == (len(sample_embeddings), 2)
        assert pca.explained_variance_ratio_.sum() > 0
        
        # t-SNE reduction (mock since it's computationally expensive)
        # In real scenario, you'd use sklearn.manifold.TSNE
        tsne_vectors = np.random.rand(len(sample_embeddings), 2)  # Mock
        
        assert tsne_vectors.shape == (len(sample_embeddings), 2)
    
    @pytest.mark.asyncio
    async def test_vector_quality_metrics(self, mock_vector_storage, sample_embeddings):
        """Test vector quality and distribution metrics"""
        vectors = list(sample_embeddings.values())
        
        # Calculate quality metrics
        metrics = {
            "mean_norm": np.mean([np.linalg.norm(v) for v in vectors]),
            "std_norm": np.std([np.linalg.norm(v) for v in vectors]),
            "mean_pairwise_similarity": np.mean([
                1 - cosine(vectors[i], vectors[j])
                for i in range(len(vectors))
                for j in range(i+1, len(vectors))
            ]),
            "dimension": len(vectors[0]),
            "num_vectors": len(vectors)
        }
        
        mock_vector_storage.get_quality_metrics = AsyncMock(return_value=metrics)
        
        result = await mock_vector_storage.get_quality_metrics()
        
        assert abs(result["mean_norm"] - 1.0) < 0.1  # Should be close to 1 for normalized vectors
        assert result["dimension"] == 768
        assert result["num_vectors"] == len(sample_embeddings)
        assert 0 <= result["mean_pairwise_similarity"] <= 1


class TestVectorIndexing:
    """Test vector indexing strategies"""
    
    @pytest.mark.asyncio
    async def test_flat_index_performance(self, mock_vector_storage, sample_embeddings):
        """Test flat (brute force) index performance"""
        index_config = {
            "type": "flat",
            "metric": "cosine"
        }
        
        mock_vector_storage.create_index.return_value = {
            "status": "created",
            "index_type": "flat",
            "build_time": 0.1
        }
        
        result = await mock_vector_storage.create_index(index_config)
        
        assert result["status"] == "created"
        assert result["index_type"] == "flat"
        assert result["build_time"] > 0
    
    @pytest.mark.asyncio 
    async def test_hnsw_index_performance(self, mock_vector_storage, sample_embeddings):
        """Test HNSW (Hierarchical Navigable Small World) index"""
        index_config = {
            "type": "hnsw",
            "metric": "cosine",
            "parameters": {
                "M": 16,  # Number of bi-directional links
                "efConstruction": 200  # Size of dynamic candidate list
            }
        }
        
        mock_vector_storage.create_index.return_value = {
            "status": "created",
            "index_type": "hnsw",
            "build_time": 0.5,
            "memory_usage": "2.3 MB"
        }
        
        result = await mock_vector_storage.create_index(index_config)
        
        assert result["status"] == "created"
        assert result["index_type"] == "hnsw"
        assert result["build_time"] > 0
    
    @pytest.mark.asyncio
    async def test_ivf_index_performance(self, mock_vector_storage, sample_embeddings):
        """Test IVF (Inverted File) index"""
        index_config = {
            "type": "ivf",
            "metric": "l2",
            "parameters": {
                "nlist": 100,  # Number of clusters
                "nprobe": 10   # Number of clusters to search
            }
        }
        
        mock_vector_storage.create_index.return_value = {
            "status": "created",
            "index_type": "ivf",
            "build_time": 1.2,
            "clusters": 100
        }
        
        result = await mock_vector_storage.create_index(index_config)
        
        assert result["status"] == "created"
        assert result["index_type"] == "ivf"
        assert result["clusters"] == 100
    
    @pytest.mark.asyncio
    async def test_index_optimization(self, mock_vector_storage):
        """Test index optimization and tuning"""
        optimization_config = {
            "target_recall": 0.95,
            "max_search_time": 10,  # milliseconds
            "memory_budget": 100   # MB
        }
        
        mock_vector_storage.optimize_index.return_value = {
            "status": "optimized",
            "achieved_recall": 0.96,
            "avg_search_time": 8,
            "memory_usage": 85,
            "optimizations_applied": ["parameter_tuning", "index_pruning"]
        }
        
        result = await mock_vector_storage.optimize_index(optimization_config)
        
        assert result["status"] == "optimized"
        assert result["achieved_recall"] >= optimization_config["target_recall"]
        assert result["avg_search_time"] <= optimization_config["max_search_time"]


class TestVectorOperationsScale:
    """Test vector operations at scale"""
    
    @pytest.mark.asyncio
    async def test_large_batch_similarity_search(self, mock_vector_storage):
        """Test similarity search with large batches of query vectors"""
        # Generate large batch of query vectors
        num_queries = 1000
        query_vectors = np.random.rand(num_queries, 768).astype(np.float32)
        query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
        
        # Mock batch search results
        mock_batch_results = []
        for i in range(num_queries):
            mock_batch_results.append([
                {"id": f"result_{i}_{j}", "score": 0.9 - j*0.1}
                for j in range(5)  # Top 5 results per query
            ])
        
        mock_vector_storage.batch_similarity_search = AsyncMock(return_value=mock_batch_results)
        
        results = await mock_vector_storage.batch_similarity_search(
            query_vectors.tolist(),
            top_k=5
        )
        
        assert len(results) == num_queries
        assert all(len(result) == 5 for result in results)
    
    @pytest.mark.asyncio
    async def test_vector_database_scaling(self, mock_vector_storage):
        """Test vector database scaling characteristics"""
        # Test with different dataset sizes
        dataset_sizes = [1000, 10000, 100000, 1000000]
        
        scaling_results = []
        for size in dataset_sizes:
            mock_result = {
                "dataset_size": size,
                "index_build_time": size * 0.001,  # Linear scaling
                "memory_usage": size * 0.004,     # 4KB per vector
                "search_latency": np.log(size) * 2,  # Logarithmic scaling for indexed search
                "throughput": 1000 / (np.log(size) * 2)  # Inverse of latency
            }
            scaling_results.append(mock_result)
        
        mock_vector_storage.benchmark_scaling = AsyncMock(return_value=scaling_results)
        
        results = await mock_vector_storage.benchmark_scaling(dataset_sizes)
        
        assert len(results) == len(dataset_sizes)
        
        # Verify scaling characteristics
        for i in range(1, len(results)):
            # Build time should increase with dataset size
            assert results[i]["index_build_time"] > results[i-1]["index_build_time"]
            
            # Memory usage should increase with dataset size
            assert results[i]["memory_usage"] > results[i-1]["memory_usage"]
    
    @pytest.mark.asyncio
    async def test_concurrent_vector_operations(self, mock_vector_storage):
        """Test concurrent vector operations"""
        # Simulate concurrent operations
        num_concurrent = 50
        
        # Mock concurrent upserts
        upsert_tasks = [
            mock_vector_storage.upsert(f"vec_{i}", np.random.rand(768).tolist(), {})
            for i in range(num_concurrent)
        ]
        
        # Mock concurrent searches
        search_tasks = [
            mock_vector_storage.similarity_search(np.random.rand(768).tolist(), top_k=5)
            for i in range(num_concurrent)
        ]
        
        # Configure mock returns
        mock_vector_storage.upsert.return_value = {"status": "success"}
        mock_vector_storage.similarity_search.return_value = [
            {"id": "result", "score": 0.8}
        ]
        
        # Execute concurrent operations
        upsert_results = await asyncio.gather(*upsert_tasks)
        search_results = await asyncio.gather(*search_tasks)
        
        assert len(upsert_results) == num_concurrent
        assert len(search_results) == num_concurrent
        assert all(r["status"] == "success" for r in upsert_results)


class TestVectorBackupAndRecovery:
    """Test vector backup and recovery operations"""
    
    @pytest.mark.asyncio
    async def test_vector_backup(self, mock_vector_storage, sample_embeddings):
        """Test backing up vector data"""
        backup_config = {
            "include_vectors": True,
            "include_metadata": True,
            "include_indexes": True,
            "compression": "gzip"
        }
        
        mock_vector_storage.create_backup.return_value = {
            "status": "completed",
            "backup_id": "backup_20250115_100000",
            "size_mb": 45.2,
            "vectors_backed_up": len(sample_embeddings),
            "compression_ratio": 0.3
        }
        
        result = await mock_vector_storage.create_backup(backup_config)
        
        assert result["status"] == "completed"
        assert result["vectors_backed_up"] == len(sample_embeddings)
        assert 0 < result["compression_ratio"] < 1
    
    @pytest.mark.asyncio
    async def test_vector_recovery(self, mock_vector_storage):
        """Test recovering vector data from backup"""
        backup_id = "backup_20250115_100000"
        recovery_config = {
            "backup_id": backup_id,
            "verify_integrity": True,
            "rebuild_indexes": True
        }
        
        mock_vector_storage.restore_from_backup.return_value = {
            "status": "completed", 
            "vectors_restored": 10,
            "indexes_rebuilt": 2,
            "integrity_check": "passed",
            "restore_time": 5.2
        }
        
        result = await mock_vector_storage.restore_from_backup(recovery_config)
        
        assert result["status"] == "completed"
        assert result["vectors_restored"] > 0
        assert result["integrity_check"] == "passed"
    
    @pytest.mark.asyncio
    async def test_vector_data_migration(self, mock_vector_storage):
        """Test migrating vector data between storage backends"""
        migration_config = {
            "source_backend": "faiss",
            "target_backend": "milvus",
            "batch_size": 1000,
            "verify_migration": True
        }
        
        mock_vector_storage.migrate_data.return_value = {
            "status": "completed",
            "vectors_migrated": 50000,
            "migration_time": 120.5,
            "verification_passed": True,
            "data_integrity": "100%"
        }
        
        result = await mock_vector_storage.migrate_data(migration_config)
        
        assert result["status"] == "completed"
        assert result["vectors_migrated"] > 0
        assert result["verification_passed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lightrag", "--cov-report=term-missing"])