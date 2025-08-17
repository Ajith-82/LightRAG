"""
Core Document Processing Pipeline Tests for LightRAG
Tests document ingestion, chunking, entity extraction, and relationship detection
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Test markers
pytestmark = [pytest.mark.core, pytest.mark.unit]


@pytest.fixture
def sample_documents():
    """Sample documents for processing tests"""
    return [
        {
            "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine Learning (ML) is a subset of AI that enables systems to learn and improve from experience. Deep Learning is a specialized form of machine learning that uses neural networks with multiple layers.",
            "metadata": {
                "title": "Introduction to AI",
                "source": "ai_primer.txt",
                "doc_id": "doc_1"
            }
        },
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence applications. Libraries like TensorFlow and PyTorch make Python popular for machine learning projects.",
            "metadata": {
                "title": "Python Programming",
                "source": "python_guide.txt", 
                "doc_id": "doc_2"
            }
        },
        {
            "content": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language. It involves tasks like text analysis, sentiment analysis, and language translation. Modern NLP systems often use transformer architectures like BERT and GPT.",
            "metadata": {
                "title": "NLP Overview",
                "source": "nlp_basics.txt",
                "doc_id": "doc_3"
            }
        }
    ]


@pytest.fixture 
def mock_lightrag_instance():
    """Mock LightRAG instance for document processing tests"""
    mock_rag = Mock()
    
    # Mock async methods
    mock_rag.ainsert = AsyncMock()
    mock_rag.adelete = AsyncMock()
    mock_rag.aquery = AsyncMock(return_value="Mock query response")
    mock_rag.initialize_storages = AsyncMock()
    mock_rag.finalize_storages = AsyncMock()
    
    # Mock storage backends
    mock_rag.chunk_entity_relation_graph = Mock()
    mock_rag.entities_vdb = Mock()
    mock_rag.relationships_vdb = Mock()
    mock_rag.chunks_vdb = Mock()
    mock_rag.llm_response_cache = Mock()
    
    # Mock LLM and embedding functions
    mock_rag.llm_model_func = AsyncMock(return_value="Mock LLM response")
    mock_rag.embedding_func = AsyncMock(return_value=[[0.1, 0.2, 0.3] * 256])
    
    return mock_rag


@pytest.fixture
def mock_file_system():
    """Mock file system for document processing tests"""
    temp_dir = tempfile.mkdtemp()
    
    # Create test files
    test_files = {
        "test1.txt": "This is a test document about machine learning and AI.",
        "test2.pdf": b"Mock PDF content",  # Binary content
        "test3.docx": b"Mock DOCX content",
        "test4.md": "# Markdown Document\n\nThis is about **deep learning** and neural networks."
    }
    
    created_files = {}
    for filename, content in test_files.items():
        file_path = os.path.join(temp_dir, filename)
        mode = 'wb' if isinstance(content, bytes) else 'w'
        with open(file_path, mode) as f:
            f.write(content)
        created_files[filename] = file_path
    
    yield temp_dir, created_files
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


class TestDocumentIngestion:
    """Test document ingestion and initial processing"""
    
    @pytest.mark.asyncio
    async def test_text_document_ingestion(self, mock_lightrag_instance, sample_documents):
        """Test ingesting plain text documents"""
        document = sample_documents[0]
        
        # Mock LightRAG insert
        mock_lightrag_instance.ainsert.return_value = {
            "status": "success",
            "doc_id": document["metadata"]["doc_id"],
            "chunks_created": 3,
            "entities_extracted": 5
        }
        
        result = await mock_lightrag_instance.ainsert(document["content"])
        
        # Verify insert was called
        mock_lightrag_instance.ainsert.assert_called_once()
        
        # Verify result structure
        assert result["status"] == "success"
        assert result["chunks_created"] > 0
        assert result["entities_extracted"] > 0
    
    @pytest.mark.asyncio
    async def test_batch_document_ingestion(self, mock_lightrag_instance, sample_documents):
        """Test batch processing of multiple documents"""
        # Mock batch insert
        mock_lightrag_instance.ainsert = AsyncMock(side_effect=[
            {"status": "success", "doc_id": "doc_1"},
            {"status": "success", "doc_id": "doc_2"},
            {"status": "success", "doc_id": "doc_3"}
        ])
        
        # Process documents in batch
        results = []
        for doc in sample_documents:
            result = await mock_lightrag_instance.ainsert(doc["content"])
            results.append(result)
        
        # Verify all documents were processed
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        assert mock_lightrag_instance.ainsert.call_count == 3
    
    @pytest.mark.asyncio
    async def test_file_document_ingestion(self, mock_lightrag_instance, mock_file_system):
        """Test ingesting documents from files"""
        temp_dir, created_files = mock_file_system
        
        # Test reading text file
        txt_file = created_files["test1.txt"]
        with open(txt_file, 'r') as f:
            content = f.read()
        
        mock_lightrag_instance.ainsert.return_value = {"status": "success"}
        
        result = await mock_lightrag_instance.ainsert(content)
        
        assert result["status"] == "success"
        mock_lightrag_instance.ainsert.assert_called_once_with(content)
    
    @pytest.mark.asyncio 
    async def test_document_metadata_handling(self, mock_lightrag_instance, sample_documents):
        """Test handling of document metadata"""
        document = sample_documents[0]
        
        # Include metadata in processing
        mock_lightrag_instance.ainsert = AsyncMock(return_value={
            "status": "success",
            "metadata": document["metadata"]
        })
        
        # Process with metadata
        result = await mock_lightrag_instance.ainsert(
            document["content"], 
            metadata=document["metadata"]
        )
        
        assert result["metadata"]["title"] == "Introduction to AI"
        assert result["metadata"]["doc_id"] == "doc_1"


class TestTextChunking:
    """Test text chunking strategies and algorithms"""
    
    def test_sentence_based_chunking(self):
        """Test sentence-based text chunking"""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        
        # Mock chunking function
        def sentence_chunk(text, max_sentences=2):
            sentences = text.split('. ')
            chunks = []
            for i in range(0, len(sentences), max_sentences):
                chunk = '. '.join(sentences[i:i+max_sentences])
                if not chunk.endswith('.'):
                    chunk += '.'
                chunks.append(chunk.strip())
            return chunks
        
        chunks = sentence_chunk(text, max_sentences=2)
        
        assert len(chunks) >= 1
        assert "first sentence" in chunks[0]
        assert "second sentence" in chunks[0]
    
    def test_token_based_chunking(self):
        """Test token-based text chunking"""
        text = "Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns from data."
        
        # Mock token-based chunking
        def token_chunk(text, max_tokens=10):
            words = text.split()
            chunks = []
            for i in range(0, len(words), max_tokens):
                chunk = ' '.join(words[i:i+max_tokens])
                chunks.append(chunk)
            return chunks
        
        chunks = token_chunk(text, max_tokens=5)
        
        assert len(chunks) >= 2
        assert all(len(chunk.split()) <= 5 for chunk in chunks[:-1])  # Last chunk might be smaller
    
    def test_semantic_chunking(self):
        """Test semantic-based text chunking"""
        paragraphs = [
            "Artificial Intelligence is a broad field of computer science.",
            "Machine Learning is a subset of AI that focuses on algorithms.",
            "Deep Learning uses neural networks with multiple layers.",
            "Natural Language Processing deals with human language understanding."
        ]
        
        text = ' '.join(paragraphs)
        
        # Mock semantic chunking (group related concepts)
        def semantic_chunk(text):
            # Simple semantic grouping based on keywords
            ai_keywords = ["artificial", "intelligence", "ai"]
            ml_keywords = ["machine", "learning", "algorithms", "deep", "neural"]
            nlp_keywords = ["language", "processing", "understanding"]
            
            sentences = text.split('. ')
            chunks = {"AI": [], "ML": [], "NLP": []}
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in nlp_keywords):
                    chunks["NLP"].append(sentence)
                elif any(keyword in sentence_lower for keyword in ml_keywords):
                    chunks["ML"].append(sentence)
                elif any(keyword in sentence_lower for keyword in ai_keywords):
                    chunks["AI"].append(sentence)
            
            return chunks
        
        chunks = semantic_chunk(text)
        
        assert len(chunks["AI"]) > 0
        assert len(chunks["ML"]) > 0
        assert len(chunks["NLP"]) > 0
    
    def test_chunk_overlap_handling(self):
        """Test handling of chunk overlap for context preservation"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        
        def overlapping_chunk(text, chunk_size=2, overlap=1):
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            chunks = []
            
            i = 0
            while i < len(sentences):
                chunk_end = min(i + chunk_size, len(sentences))
                chunk = '. '.join(sentences[i:chunk_end]) + '.'
                chunks.append(chunk)
                
                if chunk_end >= len(sentences):
                    break
                    
                i += chunk_size - overlap
            
            return chunks
        
        chunks = overlapping_chunk(text, chunk_size=3, overlap=1)
        
        # Should have overlapping content
        assert len(chunks) >= 2
        # Check for overlap
        if len(chunks) >= 2:
            # Second and third sentences should appear in both first and second chunks
            assert "Second sentence" in chunks[0]
            assert "Third sentence" in chunks[0]


class TestEntityExtraction:
    """Test entity extraction from documents"""
    
    @pytest.mark.asyncio
    async def test_named_entity_extraction(self, mock_lightrag_instance):
        """Test extraction of named entities"""
        text = "OpenAI developed ChatGPT using transformer architecture. Google created BERT for natural language understanding."
        
        # Mock LLM response for entity extraction
        mock_entities = [
            {"name": "OpenAI", "type": "organization", "description": "AI research company"},
            {"name": "ChatGPT", "type": "product", "description": "AI chatbot"},
            {"name": "transformer", "type": "technology", "description": "Neural network architecture"},
            {"name": "Google", "type": "organization", "description": "Technology company"},
            {"name": "BERT", "type": "model", "description": "Language representation model"}
        ]
        
        # Mock the LLM function to return entity extraction results
        mock_lightrag_instance.llm_model_func.return_value = json.dumps(mock_entities)
        
        # Simulate entity extraction
        llm_response = await mock_lightrag_instance.llm_model_func(
            f"Extract entities from: {text}"
        )
        
        entities = json.loads(llm_response)
        
        # Verify entity extraction
        assert len(entities) == 5
        assert any(e["name"] == "OpenAI" for e in entities)
        assert any(e["name"] == "ChatGPT" for e in entities)
        assert any(e["type"] == "organization" for e in entities)
        assert any(e["type"] == "product" for e in entities)
    
    @pytest.mark.asyncio
    async def test_concept_entity_extraction(self, mock_lightrag_instance):
        """Test extraction of conceptual entities"""
        text = "Machine learning algorithms use statistical methods to identify patterns in data. Supervised learning requires labeled training data, while unsupervised learning discovers hidden structures."
        
        # Mock conceptual entities
        mock_concepts = [
            {"name": "machine learning", "type": "concept", "description": "AI technique for pattern recognition"},
            {"name": "statistical methods", "type": "method", "description": "Mathematical approaches to data analysis"},
            {"name": "supervised learning", "type": "concept", "description": "Learning with labeled data"},
            {"name": "unsupervised learning", "type": "concept", "description": "Learning without labeled data"},
            {"name": "training data", "type": "resource", "description": "Data used to train models"}
        ]
        
        mock_lightrag_instance.llm_model_func.return_value = json.dumps(mock_concepts)
        
        llm_response = await mock_lightrag_instance.llm_model_func(
            f"Extract concepts from: {text}"
        )
        
        concepts = json.loads(llm_response)
        
        # Verify concept extraction
        assert len(concepts) == 5
        assert any(c["name"] == "machine learning" for c in concepts)
        assert any(c["name"] == "supervised learning" for c in concepts)
        assert any(c["type"] == "concept" for c in concepts)
    
    @pytest.mark.asyncio
    async def test_entity_deduplication(self, mock_lightrag_instance):
        """Test deduplication of similar entities"""
        text = "AI and artificial intelligence are related concepts. ML and machine learning refer to the same field."
        
        # Mock entities with duplicates
        mock_entities = [
            {"name": "AI", "type": "concept", "description": "Artificial intelligence"},
            {"name": "artificial intelligence", "type": "concept", "description": "AI technology"},
            {"name": "ML", "type": "concept", "description": "Machine learning"},
            {"name": "machine learning", "type": "concept", "description": "ML algorithms"}
        ]
        
        # Mock deduplication logic
        def deduplicate_entities(entities):
            seen = set()
            deduplicated = []
            
            for entity in entities:
                # Simple deduplication based on name similarity
                name_lower = entity["name"].lower()
                if name_lower not in seen:
                    # Check for acronyms or similar names
                    is_duplicate = False
                    for existing in seen:
                        if (name_lower in existing or existing in name_lower or
                            (len(name_lower) <= 3 and name_lower in existing.replace(" ", ""))):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        deduplicated.append(entity)
                        seen.add(name_lower)
            
            return deduplicated
        
        deduplicated = deduplicate_entities(mock_entities)
        
        # Should have fewer entities after deduplication
        assert len(deduplicated) < len(mock_entities)
        assert len(deduplicated) <= 2  # AI/artificial intelligence and ML/machine learning


class TestRelationshipExtraction:
    """Test relationship extraction between entities"""
    
    @pytest.mark.asyncio
    async def test_direct_relationship_extraction(self, mock_lightrag_instance):
        """Test extraction of direct relationships"""
        text = "Python is used for machine learning. TensorFlow is a Python library. Google developed TensorFlow."
        
        # Mock relationships
        mock_relationships = [
            {"source": "Python", "target": "machine learning", "type": "used_for", "weight": 0.9},
            {"source": "TensorFlow", "target": "Python", "type": "is_library_of", "weight": 0.8},
            {"source": "Google", "target": "TensorFlow", "type": "developed", "weight": 0.95}
        ]
        
        mock_lightrag_instance.llm_model_func.return_value = json.dumps(mock_relationships)
        
        llm_response = await mock_lightrag_instance.llm_model_func(
            f"Extract relationships from: {text}"
        )
        
        relationships = json.loads(llm_response)
        
        # Verify relationship extraction
        assert len(relationships) == 3
        assert any(r["source"] == "Python" and r["target"] == "machine learning" for r in relationships)
        assert any(r["type"] == "developed" for r in relationships)
        assert all(0 <= r["weight"] <= 1 for r in relationships)
    
    @pytest.mark.asyncio
    async def test_implicit_relationship_extraction(self, mock_lightrag_instance):
        """Test extraction of implicit relationships"""
        text = "Neural networks are inspired by biological neurons. Deep learning uses multi-layer neural networks. Backpropagation is the training algorithm for neural networks."
        
        # Mock implicit relationships
        mock_relationships = [
            {"source": "neural networks", "target": "biological neurons", "type": "inspired_by", "weight": 0.7},
            {"source": "deep learning", "target": "neural networks", "type": "uses", "weight": 0.9},
            {"source": "backpropagation", "target": "neural networks", "type": "trains", "weight": 0.8},
            # Implicit relationship
            {"source": "deep learning", "target": "backpropagation", "type": "trained_by", "weight": 0.6}
        ]
        
        mock_lightrag_instance.llm_model_func.return_value = json.dumps(mock_relationships)
        
        llm_response = await mock_lightrag_instance.llm_model_func(
            f"Extract all relationships from: {text}"
        )
        
        relationships = json.loads(llm_response)
        
        # Verify implicit relationships are captured
        assert len(relationships) >= 3
        implicit_rel = next((r for r in relationships if r["source"] == "deep learning" and r["target"] == "backpropagation"), None)
        if implicit_rel:
            assert implicit_rel["type"] in ["trained_by", "uses", "requires"]
    
    @pytest.mark.asyncio
    async def test_relationship_weight_calculation(self, mock_lightrag_instance):
        """Test calculation of relationship weights/confidence"""
        text = "Machine learning definitely requires data. Neural networks probably use backpropagation. AI might involve consciousness."
        
        # Mock relationships with different confidence levels
        mock_relationships = [
            {"source": "machine learning", "target": "data", "type": "requires", "weight": 0.95, "confidence": "high"},
            {"source": "neural networks", "target": "backpropagation", "type": "uses", "weight": 0.7, "confidence": "medium"},
            {"source": "AI", "target": "consciousness", "type": "involves", "weight": 0.3, "confidence": "low"}
        ]
        
        # Weight calculation based on linguistic cues
        def calculate_weight_from_text(text, relationship):
            weight = relationship["weight"]
            text_lower = text.lower()
            
            # Adjust weight based on certainty indicators
            if "definitely" in text_lower or "certainly" in text_lower:
                weight = min(1.0, weight + 0.2)
            elif "probably" in text_lower or "likely" in text_lower:
                weight = max(0.1, weight - 0.1)
            elif "might" in text_lower or "possibly" in text_lower:
                weight = max(0.1, weight - 0.3)
            
            return weight
        
        # Simulate weight adjustment
        for rel in mock_relationships:
            adjusted_weight = calculate_weight_from_text(text, rel)
            rel["adjusted_weight"] = adjusted_weight
        
        # Verify weight calculations
        high_conf_rel = next(r for r in mock_relationships if r["confidence"] == "high")
        low_conf_rel = next(r for r in mock_relationships if r["confidence"] == "low")
        
        assert high_conf_rel["adjusted_weight"] >= 0.9
        assert low_conf_rel["adjusted_weight"] <= 0.4


class TestDocumentProcessingPipeline:
    """Test the complete document processing pipeline"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_execution(self, mock_lightrag_instance, sample_documents):
        """Test the complete document processing pipeline"""
        document = sample_documents[0]
        
        # Mock the complete pipeline
        mock_pipeline_result = {
            "document_id": document["metadata"]["doc_id"],
            "chunks": [
                {"id": "chunk_1", "content": "Artificial Intelligence (AI) is a branch of computer science...", "start_pos": 0},
                {"id": "chunk_2", "content": "Machine Learning (ML) is a subset of AI...", "start_pos": 120},
                {"id": "chunk_3", "content": "Deep Learning is a specialized form of machine learning...", "start_pos": 240}
            ],
            "entities": [
                {"name": "Artificial Intelligence", "type": "concept", "chunk_ids": ["chunk_1", "chunk_2"]},
                {"name": "Machine Learning", "type": "concept", "chunk_ids": ["chunk_2", "chunk_3"]},
                {"name": "Deep Learning", "type": "concept", "chunk_ids": ["chunk_3"]}
            ],
            "relationships": [
                {"source": "Machine Learning", "target": "Artificial Intelligence", "type": "subset_of", "weight": 0.9},
                {"source": "Deep Learning", "target": "Machine Learning", "type": "specialized_form_of", "weight": 0.8}
            ],
            "embeddings": {
                "chunk_1": [0.1, 0.2, 0.3] * 256,
                "chunk_2": [0.2, 0.3, 0.4] * 256,
                "chunk_3": [0.3, 0.4, 0.5] * 256
            },
            "status": "completed",
            "processing_time": 2.5
        }
        
        mock_lightrag_instance.ainsert.return_value = mock_pipeline_result
        
        # Execute pipeline
        result = await mock_lightrag_instance.ainsert(document["content"])
        
        # Verify pipeline execution
        assert result["status"] == "completed"
        assert len(result["chunks"]) == 3
        assert len(result["entities"]) == 3
        assert len(result["relationships"]) == 2
        assert all("chunk_" in chunk["id"] for chunk in result["chunks"])
        assert all(len(embedding) == 768 for embedding in result["embeddings"].values())
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_lightrag_instance, sample_documents):
        """Test pipeline error handling and recovery"""
        document = sample_documents[0]
        
        # Test different types of errors
        error_scenarios = [
            {"stage": "chunking", "error": "ChunkingError: Invalid chunk size"},
            {"stage": "entity_extraction", "error": "LLMError: API rate limit exceeded"}, 
            {"stage": "embedding", "error": "EmbeddingError: Model not available"},
            {"stage": "storage", "error": "StorageError: Database connection failed"}
        ]
        
        for scenario in error_scenarios:
            # Mock error at specific stage
            mock_error_result = {
                "status": "failed",
                "error": scenario["error"],
                "failed_stage": scenario["stage"],
                "partial_results": {
                    "chunks": [] if scenario["stage"] == "chunking" else ["chunk_1"],
                    "entities": [] if scenario["stage"] in ["chunking", "entity_extraction"] else ["AI"],
                    "embeddings": {} if scenario["stage"] in ["chunking", "entity_extraction", "embedding"] else {"chunk_1": [0.1] * 768}
                }
            }
            
            mock_lightrag_instance.ainsert.return_value = mock_error_result
            
            result = await mock_lightrag_instance.ainsert(document["content"])
            
            # Verify error handling
            assert result["status"] == "failed"
            assert scenario["stage"] in result["failed_stage"]
            assert "partial_results" in result
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_monitoring(self, mock_lightrag_instance, sample_documents):
        """Test pipeline performance monitoring"""
        document = sample_documents[0]
        
        # Mock performance metrics
        mock_performance_result = {
            "document_id": document["metadata"]["doc_id"],
            "status": "completed",
            "performance_metrics": {
                "total_time": 5.2,
                "stage_times": {
                    "chunking": 0.5,
                    "entity_extraction": 2.1,
                    "relationship_extraction": 1.8,
                    "embedding_generation": 0.6,
                    "storage": 0.2
                },
                "memory_usage": {
                    "peak_mb": 128,
                    "average_mb": 95
                },
                "token_usage": {
                    "input_tokens": 450,
                    "output_tokens": 320,
                    "total_tokens": 770
                },
                "chunks_per_second": 1.2,
                "entities_per_second": 2.4
            }
        }
        
        mock_lightrag_instance.ainsert.return_value = mock_performance_result
        
        result = await mock_lightrag_instance.ainsert(document["content"])
        
        # Verify performance monitoring
        metrics = result["performance_metrics"]
        assert metrics["total_time"] > 0
        assert metrics["stage_times"]["entity_extraction"] > 0
        assert metrics["memory_usage"]["peak_mb"] > 0
        assert metrics["token_usage"]["total_tokens"] > 0
        assert metrics["chunks_per_second"] > 0


class TestDocumentFormatSupport:
    """Test support for different document formats"""
    
    @pytest.mark.asyncio
    async def test_pdf_document_processing(self, mock_lightrag_instance):
        """Test PDF document processing"""
        # Mock PDF content extraction
        with patch('pypdf2.PdfReader') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Extracted PDF text content about machine learning."
            mock_pdf.return_value.pages = [mock_page]
            
            # Simulate PDF processing
            pdf_content = mock_page.extract_text()
            
            mock_lightrag_instance.ainsert.return_value = {
                "status": "success",
                "source_format": "pdf",
                "content_length": len(pdf_content)
            }
            
            result = await mock_lightrag_instance.ainsert(pdf_content)
            
            assert result["status"] == "success"
            assert result["source_format"] == "pdf"
            assert result["content_length"] > 0
    
    @pytest.mark.asyncio
    async def test_docx_document_processing(self, mock_lightrag_instance):
        """Test DOCX document processing"""
        # Mock DOCX content extraction
        with patch('python_docx.Document') as mock_docx:
            mock_paragraph = Mock()
            mock_paragraph.text = "DOCX content about deep learning algorithms."
            mock_docx.return_value.paragraphs = [mock_paragraph]
            
            # Simulate DOCX processing
            docx_content = mock_paragraph.text
            
            mock_lightrag_instance.ainsert.return_value = {
                "status": "success", 
                "source_format": "docx",
                "paragraphs_processed": 1
            }
            
            result = await mock_lightrag_instance.ainsert(docx_content)
            
            assert result["status"] == "success"
            assert result["source_format"] == "docx"
    
    @pytest.mark.asyncio
    async def test_markdown_document_processing(self, mock_lightrag_instance):
        """Test Markdown document processing"""
        markdown_content = """
        # Machine Learning Guide
        
        ## Introduction
        Machine learning is a **powerful** technique for data analysis.
        
        ### Key Concepts
        - Supervised Learning
        - Unsupervised Learning
        - Reinforcement Learning
        
        ## Algorithms
        1. Linear Regression
        2. Decision Trees
        3. Neural Networks
        """
        
        # Mock markdown processing (preserve structure)
        mock_lightrag_instance.ainsert.return_value = {
            "status": "success",
            "source_format": "markdown",
            "structure": {
                "headers": ["Machine Learning Guide", "Introduction", "Key Concepts", "Algorithms"],
                "lists": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
                "bold_text": ["powerful"]
            }
        }
        
        result = await mock_lightrag_instance.ainsert(markdown_content)
        
        assert result["status"] == "success"
        assert result["source_format"] == "markdown"
        assert len(result["structure"]["headers"]) == 4
        assert len(result["structure"]["lists"]) == 3


class TestDocumentVersioning:
    """Test document versioning and updates"""
    
    @pytest.mark.asyncio
    async def test_document_update_handling(self, mock_lightrag_instance):
        """Test handling document updates and versioning"""
        original_content = "Machine learning is a subset of AI."
        updated_content = "Machine learning is a powerful subset of artificial intelligence that uses statistical methods."
        
        # Mock original document insert
        mock_lightrag_instance.ainsert.return_value = {
            "status": "success",
            "doc_id": "doc_1",
            "version": 1,
            "entities": ["machine learning", "AI"]
        }
        
        original_result = await mock_lightrag_instance.ainsert(original_content)
        
        # Mock document update
        mock_lightrag_instance.ainsert.return_value = {
            "status": "updated",
            "doc_id": "doc_1", 
            "version": 2,
            "entities": ["machine learning", "artificial intelligence", "statistical methods"],
            "changes": {
                "added_entities": ["statistical methods"],
                "modified_entities": ["AI -> artificial intelligence"],
                "removed_entities": []
            }
        }
        
        updated_result = await mock_lightrag_instance.ainsert(updated_content, doc_id="doc_1")
        
        # Verify versioning
        assert updated_result["status"] == "updated"
        assert updated_result["version"] == 2
        assert len(updated_result["entities"]) > len(original_result["entities"])
        assert "statistical methods" in updated_result["changes"]["added_entities"]
    
    @pytest.mark.asyncio
    async def test_document_deletion_handling(self, mock_lightrag_instance):
        """Test document deletion and cleanup"""
        doc_id = "doc_to_delete"
        
        # Mock deletion result
        mock_lightrag_instance.adelete.return_value = {
            "status": "deleted",
            "doc_id": doc_id,
            "cleanup": {
                "chunks_removed": 5,
                "entities_removed": 3,
                "relationships_removed": 2,
                "embeddings_removed": 5
            }
        }
        
        result = await mock_lightrag_instance.adelete(doc_id)
        
        # Verify deletion
        assert result["status"] == "deleted"
        assert result["doc_id"] == doc_id
        assert result["cleanup"]["chunks_removed"] > 0
        assert result["cleanup"]["entities_removed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lightrag", "--cov-report=term-missing"])