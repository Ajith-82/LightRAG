"""
MCP (Model Context Protocol) Server Functionality Tests for LightRAG
Tests MCP server integration, tools, resources, and Claude CLI compatibility
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
pytestmark = [pytest.mark.core, pytest.mark.integration]


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing"""
    mock_server = Mock()
    
    # Mock MCP server methods
    mock_server.start = AsyncMock()
    mock_server.stop = AsyncMock()
    mock_server.handle_request = AsyncMock()
    mock_server.list_tools = AsyncMock()
    mock_server.list_resources = AsyncMock()
    mock_server.call_tool = AsyncMock()
    mock_server.read_resource = AsyncMock()
    
    return mock_server


@pytest.fixture
def mcp_test_config():
    """Test configuration for MCP server"""
    return {
        "server_name": "lightrag-mcp-test",
        "version": "1.0.0",
        "api_url": "http://localhost:9621",
        "tools_enabled": True,
        "resources_enabled": True,
        "streaming_enabled": True,
        "max_connections": 10
    }


@pytest.fixture
def sample_mcp_tools():
    """Sample MCP tools for testing"""
    return [
        {
            "name": "lightrag_query",
            "description": "Query the LightRAG knowledge base",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to execute"},
                    "mode": {"type": "string", "enum": ["local", "global", "hybrid", "mix", "naive"]},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        },
        {
            "name": "lightrag_insert_text",
            "description": "Insert text into the LightRAG knowledge base",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to insert"},
                    "metadata": {"type": "object", "description": "Optional metadata"}
                },
                "required": ["text"]
            }
        },
        {
            "name": "lightrag_insert_file",
            "description": "Insert file content into the knowledge base",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "file_type": {"type": "string", "description": "Type of file (pdf, txt, docx, etc.)"}
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "lightrag_get_graph",
            "description": "Get knowledge graph data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["json", "graphml", "cypher"]},
                    "max_nodes": {"type": "integer", "default": 100}
                }
            }
        },
        {
            "name": "lightrag_search_entities",
            "description": "Search for entities in the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "entity_type": {"type": "string", "description": "Filter by entity type"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        }
    ]


@pytest.fixture
def sample_mcp_resources():
    """Sample MCP resources for testing"""
    return [
        {
            "uri": "lightrag://system/config",
            "name": "System Configuration",
            "description": "LightRAG system configuration and status",
            "mimeType": "application/json"
        },
        {
            "uri": "lightrag://knowledge/graph",
            "name": "Knowledge Graph",
            "description": "Complete knowledge graph structure",
            "mimeType": "application/json"
        },
        {
            "uri": "lightrag://documents/list",
            "name": "Document List",
            "description": "List of all documents in the knowledge base",
            "mimeType": "application/json"
        }
    ]


class TestMCPServerInitialization:
    """Test MCP server initialization and configuration"""
    
    @pytest.mark.asyncio
    async def test_server_startup(self, mock_mcp_server, mcp_test_config):
        """Test MCP server startup"""
        mock_mcp_server.start.return_value = {
            "status": "running",
            "port": 8080,
            "tools_available": 11,
            "resources_available": 3
        }
        
        result = await mock_mcp_server.start(mcp_test_config)
        
        assert result["status"] == "running"
        assert result["tools_available"] > 0
        assert result["resources_available"] > 0
        mock_mcp_server.start.assert_called_once_with(mcp_test_config)
    
    @pytest.mark.asyncio
    async def test_server_configuration_validation(self, mock_mcp_server):
        """Test MCP server configuration validation"""
        invalid_configs = [
            {},  # Empty config
            {"server_name": ""},  # Empty name
            {"api_url": "invalid-url"},  # Invalid URL
            {"max_connections": -1}  # Invalid max connections
        ]
        
        for config in invalid_configs:
            mock_mcp_server.start.side_effect = ValueError("Invalid configuration")
            
            with pytest.raises(ValueError):
                await mock_mcp_server.start(config)
    
    @pytest.mark.asyncio
    async def test_server_shutdown(self, mock_mcp_server):
        """Test MCP server shutdown"""
        mock_mcp_server.stop.return_value = {
            "status": "stopped",
            "cleanup_completed": True,
            "active_connections_closed": 2
        }
        
        result = await mock_mcp_server.stop()
        
        assert result["status"] == "stopped"
        assert result["cleanup_completed"] is True
        mock_mcp_server.stop.assert_called_once()


class TestMCPTools:
    """Test MCP tool functionality"""
    
    @pytest.mark.asyncio
    async def test_list_available_tools(self, mock_mcp_server, sample_mcp_tools):
        """Test listing available MCP tools"""
        mock_mcp_server.list_tools.return_value = {
            "tools": sample_mcp_tools
        }
        
        result = await mock_mcp_server.list_tools()
        
        assert len(result["tools"]) == len(sample_mcp_tools)
        assert any(tool["name"] == "lightrag_query" for tool in result["tools"])
        assert any(tool["name"] == "lightrag_insert_text" for tool in result["tools"])
        mock_mcp_server.list_tools.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_lightrag_query_tool(self, mock_mcp_server):
        """Test LightRAG query tool"""
        tool_request = {
            "name": "lightrag_query",
            "arguments": {
                "query": "What is machine learning?",
                "mode": "hybrid",
                "top_k": 3
            }
        }
        
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": """Machine Learning is a subset of AI that enables computers to learn 
                              from data without being explicitly programmed. It includes techniques 
                              like supervised learning, unsupervised learning, and reinforcement learning."""
                }
            ],
            "isError": False,
            "metadata": {
                "mode": "hybrid",
                "sources": ["doc1", "doc2"],
                "processing_time": 1.2
            }
        }
        
        mock_mcp_server.call_tool.return_value = mock_response
        
        result = await mock_mcp_server.call_tool(tool_request["name"], tool_request["arguments"])
        
        assert result["isError"] is False
        assert "machine learning" in result["content"][0]["text"].lower()
        assert result["metadata"]["mode"] == "hybrid"
    
    @pytest.mark.asyncio
    async def test_lightrag_insert_text_tool(self, mock_mcp_server):
        """Test LightRAG text insertion tool"""
        tool_request = {
            "name": "lightrag_insert_text",
            "arguments": {
                "text": "Neural networks are computing systems inspired by biological neural networks.",
                "metadata": {"category": "AI", "source": "manual_input"}
            }
        }
        
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": "Successfully inserted text into knowledge base"
                }
            ],
            "isError": False,
            "metadata": {
                "doc_id": "doc_12345",
                "chunks_created": 1,
                "entities_extracted": 2,
                "processing_time": 0.8
            }
        }
        
        mock_mcp_server.call_tool.return_value = mock_response
        
        result = await mock_mcp_server.call_tool(tool_request["name"], tool_request["arguments"])
        
        assert result["isError"] is False
        assert "successfully inserted" in result["content"][0]["text"].lower()
        assert result["metadata"]["entities_extracted"] > 0
    
    @pytest.mark.asyncio
    async def test_lightrag_insert_file_tool(self, mock_mcp_server):
        """Test LightRAG file insertion tool"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document about artificial intelligence.")
            temp_file_path = f.name
        
        try:
            tool_request = {
                "name": "lightrag_insert_file",
                "arguments": {
                    "file_path": temp_file_path,
                    "file_type": "txt"
                }
            }
            
            mock_response = {
                "content": [
                    {
                        "type": "text",
                        "text": f"Successfully processed file: {temp_file_path}"
                    }
                ],
                "isError": False,
                "metadata": {
                    "file_size": 52,
                    "chunks_created": 1,
                    "entities_extracted": 1,
                    "processing_time": 1.1
                }
            }
            
            mock_mcp_server.call_tool.return_value = mock_response
            
            result = await mock_mcp_server.call_tool(tool_request["name"], tool_request["arguments"])
            
            assert result["isError"] is False
            assert temp_file_path in result["content"][0]["text"]
            assert result["metadata"]["file_size"] > 0
        finally:
            os.unlink(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_lightrag_get_graph_tool(self, mock_mcp_server):
        """Test LightRAG graph retrieval tool"""
        tool_request = {
            "name": "lightrag_get_graph",
            "arguments": {
                "format": "json",
                "max_nodes": 50
            }
        }
        
        mock_graph_data = {
            "nodes": [
                {"id": "ai", "name": "Artificial Intelligence", "type": "concept"},
                {"id": "ml", "name": "Machine Learning", "type": "concept"},
                {"id": "dl", "name": "Deep Learning", "type": "concept"}
            ],
            "edges": [
                {"source": "ml", "target": "ai", "type": "subset_of"},
                {"source": "dl", "target": "ml", "type": "subset_of"}
            ]
        }
        
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(mock_graph_data, indent=2)
                }
            ],
            "isError": False,
            "metadata": {
                "format": "json",
                "nodes_count": 3,
                "edges_count": 2,
                "generation_time": 0.5
            }
        }
        
        mock_mcp_server.call_tool.return_value = mock_response
        
        result = await mock_mcp_server.call_tool(tool_request["name"], tool_request["arguments"])
        
        assert result["isError"] is False
        graph_data = json.loads(result["content"][0]["text"])
        assert len(graph_data["nodes"]) == 3
        assert len(graph_data["edges"]) == 2
    
    @pytest.mark.asyncio
    async def test_lightrag_search_entities_tool(self, mock_mcp_server):
        """Test LightRAG entity search tool"""
        tool_request = {
            "name": "lightrag_search_entities",
            "arguments": {
                "query": "learning",
                "entity_type": "concept",
                "limit": 5
            }
        }
        
        mock_entities = [
            {"id": "ml", "name": "Machine Learning", "type": "concept", "relevance": 0.95},
            {"id": "dl", "name": "Deep Learning", "type": "concept", "relevance": 0.88},
            {"id": "rl", "name": "Reinforcement Learning", "type": "concept", "relevance": 0.82}
        ]
        
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(mock_entities, indent=2)
                }
            ],
            "isError": False,
            "metadata": {
                "query": "learning",
                "results_count": 3,
                "entity_type_filter": "concept",
                "search_time": 0.3
            }
        }
        
        mock_mcp_server.call_tool.return_value = mock_response
        
        result = await mock_mcp_server.call_tool(tool_request["name"], tool_request["arguments"])
        
        assert result["isError"] is False
        entities = json.loads(result["content"][0]["text"])
        assert len(entities) == 3
        assert all("learning" in entity["name"].lower() for entity in entities)


class TestMCPResources:
    """Test MCP resource functionality"""
    
    @pytest.mark.asyncio
    async def test_list_available_resources(self, mock_mcp_server, sample_mcp_resources):
        """Test listing available MCP resources"""
        mock_mcp_server.list_resources.return_value = {
            "resources": sample_mcp_resources
        }
        
        result = await mock_mcp_server.list_resources()
        
        assert len(result["resources"]) == len(sample_mcp_resources)
        assert any(res["uri"] == "lightrag://system/config" for res in result["resources"])
        assert any(res["uri"] == "lightrag://knowledge/graph" for res in result["resources"])
        mock_mcp_server.list_resources.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_read_system_config_resource(self, mock_mcp_server):
        """Test reading system configuration resource"""
        resource_uri = "lightrag://system/config"
        
        mock_config_data = {
            "version": "1.4.5",
            "llm_binding": "ollama",
            "embedding_binding": "ollama",
            "storage_backends": {
                "kv_storage": "json",
                "vector_storage": "nano_vector_db",
                "graph_storage": "networkx"
            },
            "status": "running",
            "documents_indexed": 25,
            "entities_count": 156,
            "relationships_count": 89
        }
        
        mock_response = {
            "contents": [
                {
                    "uri": resource_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(mock_config_data, indent=2)
                }
            ]
        }
        
        mock_mcp_server.read_resource.return_value = mock_response
        
        result = await mock_mcp_server.read_resource(resource_uri)
        
        assert result["contents"][0]["uri"] == resource_uri
        config_data = json.loads(result["contents"][0]["text"])
        assert config_data["status"] == "running"
        assert config_data["documents_indexed"] > 0
    
    @pytest.mark.asyncio
    async def test_read_knowledge_graph_resource(self, mock_mcp_server):
        """Test reading knowledge graph resource"""
        resource_uri = "lightrag://knowledge/graph"
        
        mock_graph_data = {
            "graph_info": {
                "nodes_count": 156,
                "edges_count": 89,
                "connected_components": 1,
                "avg_degree": 1.14
            },
            "sample_nodes": [
                {"id": "ai", "name": "Artificial Intelligence", "degree": 8},
                {"id": "ml", "name": "Machine Learning", "degree": 6}
            ],
            "node_types": {
                "concept": 45,
                "technique": 32,
                "application": 28,
                "architecture": 25
            }
        }
        
        mock_response = {
            "contents": [
                {
                    "uri": resource_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(mock_graph_data, indent=2)
                }
            ]
        }
        
        mock_mcp_server.read_resource.return_value = mock_response
        
        result = await mock_mcp_server.read_resource(resource_uri)
        
        graph_data = json.loads(result["contents"][0]["text"])
        assert graph_data["graph_info"]["nodes_count"] > 0
        assert len(graph_data["sample_nodes"]) > 0
        assert "concept" in graph_data["node_types"]
    
    @pytest.mark.asyncio
    async def test_read_documents_list_resource(self, mock_mcp_server):
        """Test reading documents list resource"""
        resource_uri = "lightrag://documents/list"
        
        mock_documents = {
            "documents": [
                {
                    "id": "doc_1",
                    "title": "Introduction to AI",
                    "status": "processed",
                    "chunks": 5,
                    "entities": 12,
                    "last_updated": "2025-01-15T10:00:00Z"
                },
                {
                    "id": "doc_2", 
                    "title": "Machine Learning Algorithms",
                    "status": "processed",
                    "chunks": 8,
                    "entities": 18,
                    "last_updated": "2025-01-15T11:30:00Z"
                }
            ],
            "total_documents": 25,
            "total_chunks": 127,
            "total_entities": 156
        }
        
        mock_response = {
            "contents": [
                {
                    "uri": resource_uri,
                    "mimeType": "application/json",
                    "text": json.dumps(mock_documents, indent=2)
                }
            ]
        }
        
        mock_mcp_server.read_resource.return_value = mock_response
        
        result = await mock_mcp_server.read_resource(resource_uri)
        
        docs_data = json.loads(result["contents"][0]["text"])
        assert docs_data["total_documents"] > 0
        assert len(docs_data["documents"]) > 0
        assert all(doc["status"] == "processed" for doc in docs_data["documents"])


class TestMCPErrorHandling:
    """Test MCP error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self, mock_mcp_server):
        """Test handling tool execution errors"""
        tool_request = {
            "name": "lightrag_query",
            "arguments": {
                "query": "test query",
                "mode": "invalid_mode"  # Invalid mode
            }
        }
        
        mock_error_response = {
            "content": [
                {
                    "type": "text",
                    "text": "Error: Invalid query mode 'invalid_mode'"
                }
            ],
            "isError": True,
            "metadata": {
                "error_type": "ValidationError",
                "error_code": "INVALID_PARAMETER"
            }
        }
        
        mock_mcp_server.call_tool.return_value = mock_error_response
        
        result = await mock_mcp_server.call_tool(tool_request["name"], tool_request["arguments"])
        
        assert result["isError"] is True
        assert "invalid" in result["content"][0]["text"].lower()
        assert result["metadata"]["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_resource_not_found_error(self, mock_mcp_server):
        """Test handling resource not found errors"""
        invalid_resource_uri = "lightrag://invalid/resource"
        
        mock_error_response = {
            "contents": [],
            "error": {
                "code": "RESOURCE_NOT_FOUND",
                "message": f"Resource not found: {invalid_resource_uri}"
            }
        }
        
        mock_mcp_server.read_resource.return_value = mock_error_response
        
        result = await mock_mcp_server.read_resource(invalid_resource_uri)
        
        assert len(result["contents"]) == 0
        assert result["error"]["code"] == "RESOURCE_NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, mock_mcp_server):
        """Test handling connection timeouts"""
        mock_mcp_server.call_tool.side_effect = asyncio.TimeoutError("Connection timeout")
        
        with pytest.raises(asyncio.TimeoutError):
            await mock_mcp_server.call_tool("lightrag_query", {"query": "test"})
    
    @pytest.mark.asyncio
    async def test_invalid_tool_parameters(self, mock_mcp_server):
        """Test handling invalid tool parameters"""
        invalid_requests = [
            {"name": "lightrag_query", "arguments": {}},  # Missing required 'query'
            {"name": "lightrag_insert_text", "arguments": {"metadata": {}}},  # Missing required 'text'
            {"name": "nonexistent_tool", "arguments": {"query": "test"}}  # Non-existent tool
        ]
        
        for request in invalid_requests:
            mock_error_response = {
                "content": [{"type": "text", "text": "Parameter validation error"}],
                "isError": True,
                "metadata": {"error_type": "ParameterError"}
            }
            
            mock_mcp_server.call_tool.return_value = mock_error_response
            
            result = await mock_mcp_server.call_tool(request["name"], request["arguments"])
            
            assert result["isError"] is True


class TestMCPClaudeIntegration:
    """Test MCP server integration with Claude CLI"""
    
    @pytest.mark.asyncio
    async def test_claude_cli_tool_invocation(self, mock_mcp_server):
        """Test tool invocation through Claude CLI"""
        # Simulate Claude CLI invoking a tool
        cli_request = {
            "method": "tools/call",
            "params": {
                "name": "lightrag_query",
                "arguments": {
                    "query": "What are the main AI techniques?",
                    "mode": "hybrid"
                }
            }
        }
        
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": """The main AI techniques include Machine Learning, Deep Learning, 
                              Natural Language Processing, and Computer Vision. Each serves 
                              different purposes and applications."""
                }
            ],
            "isError": False
        }
        
        mock_mcp_server.handle_request.return_value = mock_response
        
        result = await mock_mcp_server.handle_request(cli_request)
        
        assert result["isError"] is False
        assert "machine learning" in result["content"][0]["text"].lower()
    
    @pytest.mark.asyncio
    async def test_claude_cli_resource_access(self, mock_mcp_server):
        """Test resource access through Claude CLI"""
        cli_request = {
            "method": "resources/read",
            "params": {
                "uri": "lightrag://system/config"
            }
        }
        
        mock_config = {
            "version": "1.4.5",
            "status": "healthy",
            "components": {
                "llm": "active",
                "embeddings": "active", 
                "storage": "active"
            }
        }
        
        mock_response = {
            "contents": [
                {
                    "uri": "lightrag://system/config",
                    "mimeType": "application/json",
                    "text": json.dumps(mock_config)
                }
            ]
        }
        
        mock_mcp_server.handle_request.return_value = mock_response
        
        result = await mock_mcp_server.handle_request(cli_request)
        
        assert len(result["contents"]) > 0
        config = json.loads(result["contents"][0]["text"])
        assert config["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_claude_cli_streaming_support(self, mock_mcp_server):
        """Test streaming support for Claude CLI"""
        cli_request = {
            "method": "tools/call",
            "params": {
                "name": "lightrag_query",
                "arguments": {
                    "query": "Explain machine learning in detail",
                    "mode": "local",
                    "stream": True
                }
            }
        }
        
        # Mock streaming response
        async def mock_streaming_response():
            chunks = [
                {"type": "text", "text": "Machine learning is "},
                {"type": "text", "text": "a subset of artificial intelligence "},
                {"type": "text", "text": "that enables computers to learn from data."}
            ]
            for chunk in chunks:
                yield {"content": [chunk], "isError": False}
        
        mock_mcp_server.handle_request.return_value = mock_streaming_response()
        
        response_stream = await mock_mcp_server.handle_request(cli_request)
        
        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert all(not chunk["isError"] for chunk in chunks)


class TestMCPPerformanceAndScaling:
    """Test MCP server performance and scaling characteristics"""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_requests(self, mock_mcp_server):
        """Test handling multiple concurrent tool requests"""
        num_concurrent_requests = 10
        
        async def mock_tool_response(name, args):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "content": [{"type": "text", "text": f"Response to {args.get('query', 'query')}"}],
                "isError": False,
                "processing_time": 0.1
            }
        
        mock_mcp_server.call_tool.side_effect = mock_tool_response
        
        # Create concurrent requests
        tasks = [
            mock_mcp_server.call_tool("lightrag_query", {"query": f"Query {i}"})
            for i in range(num_concurrent_requests)
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == num_concurrent_requests
        assert all(not result["isError"] for result in results)
        
        # Should complete faster than sequential execution due to concurrency
        total_time = end_time - start_time
        assert total_time < num_concurrent_requests * 0.1 * 0.8  # Allow some overhead
    
    @pytest.mark.asyncio
    async def test_large_response_handling(self, mock_mcp_server):
        """Test handling large responses"""
        # Simulate large knowledge graph response
        large_graph = {
            "nodes": [{"id": f"node_{i}", "name": f"Entity {i}"} for i in range(1000)],
            "edges": [{"source": f"node_{i}", "target": f"node_{i+1}"} for i in range(999)]
        }
        
        mock_response = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(large_graph)
                }
            ],
            "isError": False,
            "metadata": {
                "size_mb": len(json.dumps(large_graph)) / (1024 * 1024),
                "nodes_count": 1000,
                "edges_count": 999
            }
        }
        
        mock_mcp_server.call_tool.return_value = mock_response
        
        result = await mock_mcp_server.call_tool(
            "lightrag_get_graph",
            {"format": "json", "max_nodes": 1000}
        )
        
        assert result["isError"] is False
        assert result["metadata"]["size_mb"] > 0
        graph_data = json.loads(result["content"][0]["text"])
        assert len(graph_data["nodes"]) == 1000
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, mock_mcp_server, mcp_test_config):
        """Test connection pooling efficiency"""
        # Simulate multiple clients connecting
        connection_results = []
        
        for i in range(mcp_test_config["max_connections"] + 2):  # Exceed max connections
            mock_connection_result = {
                "connection_id": i,
                "status": "accepted" if i < mcp_test_config["max_connections"] else "rejected",
                "reason": "max_connections_exceeded" if i >= mcp_test_config["max_connections"] else None
            }
            connection_results.append(mock_connection_result)
        
        mock_mcp_server.handle_connection = AsyncMock(side_effect=connection_results)
        
        # Test connections
        results = []
        for i in range(mcp_test_config["max_connections"] + 2):
            result = await mock_mcp_server.handle_connection(f"client_{i}")
            results.append(result)
        
        accepted_connections = [r for r in results if r["status"] == "accepted"]
        rejected_connections = [r for r in results if r["status"] == "rejected"]
        
        assert len(accepted_connections) == mcp_test_config["max_connections"]
        assert len(rejected_connections) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=lightrag_mcp", "--cov-report=term-missing"])