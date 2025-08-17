"""
Direct library client for LightRAG MCP integration.

Provides direct access to LightRAG library functions without going
through the REST API. Used when MCP server runs in the same environment
as LightRAG core library.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..config import LightRAGMCPConfig
from ..utils import MCPError

logger = logging.getLogger("lightrag-mcp.direct_client")


class DirectClientError(Exception):
    """Exception raised by direct client operations."""

    pass


class LightRAGDirectClient:
    """Direct library interface for LightRAG."""

    def __init__(self, config: LightRAGMCPConfig):
        self.config = config
        self._lightrag = None
        self._initialized = False
        self._document_tracker = {}  # Simple in-memory document tracking
        self._query_stats = {
            "total_queries": 0,
            "queries_by_mode": {},
            "total_response_time": 0.0,
        }

    async def __aenter__(self) -> "LightRAGDirectClient":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._lightrag:
            try:
                if hasattr(self._lightrag, "finalize_storages"):
                    await self._lightrag.finalize_storages()
            except Exception as e:
                logger.warning(f"Error finalizing LightRAG storages: {e}")
            self._lightrag = None
            self._initialized = False

    async def _ensure_initialized(self):
        """Ensure LightRAG instance is initialized."""
        if self._initialized and self._lightrag:
            return

        try:
            # Import LightRAG here to avoid dependency issues
            from lightrag import LightRAG

            # Determine working directory
            working_dir = self.config.lightrag_working_dir or "./rag_storage"

            logger.info(f"Initializing LightRAG with working directory: {working_dir}")

            # Create LightRAG instance
            self._lightrag = LightRAG(
                working_dir=working_dir,
                # Add other configuration parameters as needed
            )

            # Initialize storages (pipeline status initialization might not be needed)
            if hasattr(self._lightrag, "initialize_storages"):
                await self._lightrag.initialize_storages()

            self._initialized = True
            logger.info("LightRAG direct client initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import LightRAG: {e}")
            raise DirectClientError(f"LightRAG library not available: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            raise DirectClientError(f"LightRAG initialization failed: {e}")

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous function in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    # Query operations
    async def query(self, query: str, mode: str = "hybrid", **params) -> Dict[str, Any]:
        """Execute a RAG query using direct library access."""
        await self._ensure_initialized()

        start_time = time.time()
        try:
            logger.info(f"Executing direct query: {query[:100]}... (mode: {mode})")

            # Execute query using LightRAG
            result = await self._lightrag.aquery(query, param=params, mode=mode)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._query_stats["total_queries"] += 1
            self._query_stats["queries_by_mode"][mode] = self._query_stats["queries_by_mode"].get(mode, 0) + 1
            self._query_stats["total_response_time"] += processing_time

            # Format response to match API format
            return {
                "response": result,
                "mode": mode,
                "metadata": {
                    "processing_time": processing_time,
                    "entities_used": self._estimate_entities_used(result),
                    "relations_used": self._estimate_relations_used(result),
                    "chunks_used": self._estimate_chunks_used(result),
                    "token_usage": {
                        "prompt_tokens": len(query.split()) * 1.3,  # Rough estimate
                        "completion_tokens": len(str(result).split()) * 1.3,
                        "total_tokens": (len(query) + len(str(result))) * 1.3,
                    },
                },
                "sources": self._extract_sources(result),
            }

        except Exception as e:
            logger.error(f"Direct query failed: {e}")
            raise MCPError("QUERY_FAILED", f"Query execution failed: {e}")
    
    def _estimate_entities_used(self, result: str) -> int:
        """Estimate number of entities used in query result."""
        # Simple heuristic: count capitalized words that might be entities
        import re
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', result)
        return len(set(entities))
    
    def _estimate_relations_used(self, result: str) -> int:
        """Estimate number of relations used in query result."""
        # Simple heuristic: count common relation indicators
        relation_indicators = ['related to', 'connected to', 'part of', 'works with', 'associated with']
        count = sum(result.lower().count(indicator) for indicator in relation_indicators)
        return count
    
    def _estimate_chunks_used(self, result: str) -> int:
        """Estimate number of chunks used in query result."""
        # Simple heuristic: assume 1 chunk per 200 words
        word_count = len(result.split())
        return max(1, word_count // 200)
    
    def _extract_sources(self, result: str) -> List[Dict[str, Any]]:
        """Extract source information from query result."""
        # Simple implementation - in reality this would need integration with LightRAG's source tracking
        return []

    async def stream_query(
        self, query: str, mode: str = "hybrid", **params
    ) -> AsyncIterator[str]:
        """Execute a streaming RAG query with chunked response."""
        await self._ensure_initialized()
        
        try:
            logger.info(f"Executing streaming query: {query[:100]}... (mode: {mode})")
            
            # Execute the query
            result = await self._lightrag.aquery(query, param=params, mode=mode)
            
            # Stream the result in chunks to simulate streaming
            result_str = str(result)
            chunk_size = 100  # Characters per chunk
            
            for i in range(0, len(result_str), chunk_size):
                chunk = result_str[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.01)  # Small delay to simulate real streaming
                
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            raise MCPError("QUERY_FAILED", f"Streaming query execution failed: {e}")

    # Document operations
    async def insert_text(
        self, text: str, title: str = "", metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Insert text document using direct library access."""
        await self._ensure_initialized()

        try:
            logger.info(f"Inserting text document: {title or 'Untitled'}")

            # Generate unique document ID
            doc_id = f"direct_{abs(hash(text + title + str(time.time())))}"
            
            # Insert text using LightRAG
            await self._lightrag.ainsert(text)
            
            # Track document
            doc_info = {
                "id": doc_id,
                "title": title or "Untitled",
                "content": text[:200] + "..." if len(text) > 200 else text,
                "metadata": metadata or {},
                "status": "processed",
                "created_at": time.time(),
                "size": len(text),
                "type": "text"
            }
            self._document_tracker[doc_id] = doc_info

            # Estimate processing info
            word_count = len(text.split())
            estimated_chunks = max(1, word_count // 200)
            estimated_entities = len(set(word for word in text.split() if word[0].isupper()))
            
            # Return formatted response
            return {
                "document_id": doc_id,
                "status": "processed",
                "message": "Document inserted successfully",
                "processing_info": {
                    "chunks_created": estimated_chunks,
                    "entities_extracted": estimated_entities,
                    "relationships_created": estimated_entities // 2,  # Rough estimate
                    "text_length": len(text),
                    "word_count": word_count,
                },
            }

        except Exception as e:
            logger.error(f"Direct text insertion failed: {e}")
            raise MCPError("PROCESSING_FAILED", f"Text insertion failed: {e}")

    async def insert_file(self, file_path: str, **options) -> Dict[str, Any]:
        """Insert file document using direct library access."""
        await self._ensure_initialized()

        # Validate file
        path = Path(file_path)
        if not path.exists():
            raise MCPError("FILE_NOT_FOUND", f"File not found: {file_path}")

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise MCPError(
                "FILE_TOO_LARGE",
                f"File size {file_size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB",
            )

        try:
            logger.info(f"Inserting file: {file_path}")

            # Read file content
            content = path.read_text(encoding="utf-8")

            # Insert using text insertion
            return await self.insert_text(content, title=path.name)

        except UnicodeDecodeError:
            logger.error(f"Failed to decode file: {file_path}")
            raise MCPError(
                "UNSUPPORTED_FORMAT", f"Cannot decode file as text: {file_path}"
            )

        except Exception as e:
            logger.error(f"Direct file insertion failed: {e}")
            raise MCPError("PROCESSING_FAILED", f"File insertion failed: {e}")

    async def list_documents(
        self, status_filter: str = "", limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """List documents with tracking support."""
        try:
            # Filter documents based on status
            docs = list(self._document_tracker.values())
            
            if status_filter:
                docs = [doc for doc in docs if doc["status"] == status_filter]
            
            # Sort by creation time (newest first)
            docs.sort(key=lambda x: x["created_at"], reverse=True)
            
            # Apply pagination
            total = len(docs)
            start_idx = offset
            end_idx = offset + limit
            paginated_docs = docs[start_idx:end_idx]
            
            # Format documents for response
            formatted_docs = []
            for doc in paginated_docs:
                formatted_docs.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "status": doc["status"],
                    "created_at": doc["created_at"],
                    "size": doc["size"],
                    "type": doc["type"],
                    "preview": doc["content"]
                })
            
            return {
                "documents": formatted_docs,
                "total": total,
                "has_more": end_idx < total,
                "limit": limit,
                "offset": offset,
            }
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise MCPError("LIST_FAILED", f"Document listing failed: {e}")

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get specific document by ID."""
        try:
            if document_id not in self._document_tracker:
                raise MCPError("DOCUMENT_NOT_FOUND", f"Document {document_id} not found")
            
            doc = self._document_tracker[document_id]
            return {
                "document": {
                    "id": doc["id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "status": doc["status"],
                    "created_at": doc["created_at"],
                    "size": doc["size"],
                    "type": doc["type"]
                }
            }
            
        except MCPError:
            raise
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise MCPError("RETRIEVAL_FAILED", f"Document retrieval failed: {e}")

    async def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents from tracking (note: data remains in LightRAG storage)."""
        try:
            deleted = []
            not_found = []
            
            for doc_id in document_ids:
                if doc_id in self._document_tracker:
                    del self._document_tracker[doc_id]
                    deleted.append(doc_id)
                else:
                    not_found.append(doc_id)
            
            return {
                "deleted": deleted,
                "not_found": not_found,
                "message": f"Deleted {len(deleted)} documents from tracking. Note: Data remains in LightRAG storage.",
                "warning": "Direct mode deletion only removes tracking information, not actual data from LightRAG storage."
            }
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise MCPError("DELETION_FAILED", f"Document deletion failed: {e}")

    # Graph operations
    async def get_graph(self, **params) -> Dict[str, Any]:
        """Get knowledge graph data by accessing LightRAG storage."""
        await self._ensure_initialized()
        
        try:
            max_nodes = params.get("max_nodes", 100)
            max_edges = params.get("max_edges", 200)
            
            # Access LightRAG's graph storage directly
            nodes = []
            edges = []
            
            # Try to extract graph data from storage
            if hasattr(self._lightrag, "kg_storage"):
                try:
                    # Get all entities (nodes)
                    entity_data = await self._lightrag.kg_storage.get("entities")
                    if entity_data:
                        entities = json.loads(entity_data) if isinstance(entity_data, str) else entity_data
                        for entity_id, entity_info in list(entities.items())[:max_nodes]:
                            nodes.append({
                                "id": entity_id,
                                "label": entity_info.get("name", entity_id),
                                "type": entity_info.get("type", "entity"),
                                "description": entity_info.get("description", ""),
                                "properties": entity_info
                            })
                    
                    # Get all relationships (edges)
                    relation_data = await self._lightrag.kg_storage.get("relationships")
                    if relation_data:
                        relationships = json.loads(relation_data) if isinstance(relation_data, str) else relation_data
                        for rel_id, rel_info in list(relationships.items())[:max_edges]:
                            edges.append({
                                "id": rel_id,
                                "source": rel_info.get("source", ""),
                                "target": rel_info.get("target", ""),
                                "label": rel_info.get("type", ""),
                                "description": rel_info.get("description", ""),
                                "properties": rel_info
                            })
                
                except Exception as e:
                    logger.warning(f"Error accessing graph storage: {e}")
            
            # Generate statistics
            node_types = {}
            edge_types = {}
            
            for node in nodes:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            for edge in edges:
                edge_type = edge.get("label", "unknown")
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            return {
                "nodes": nodes,
                "edges": edges,
                "statistics": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "node_types": node_types,
                    "edge_types": edge_types,
                },
                "parameters": {
                    "max_nodes": max_nodes,
                    "max_edges": max_edges,
                    "returned_nodes": len(nodes),
                    "returned_edges": len(edges)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return {
                "nodes": [],
                "edges": [],
                "statistics": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "node_types": {},
                    "edge_types": {},
                },
                "error": str(e),
                "message": "Graph extraction failed in direct mode",
            }

    async def search_entities(
        self, query: str, limit: int = 20, **params
    ) -> Dict[str, Any]:
        """Search entities in the knowledge graph."""
        await self._ensure_initialized()
        
        try:
            logger.info(f"Searching entities for query: {query}")
            
            entities = []
            query_lower = query.lower()
            
            # Access LightRAG's graph storage to search entities
            if hasattr(self._lightrag, "kg_storage"):
                try:
                    entity_data = await self._lightrag.kg_storage.get("entities")
                    if entity_data:
                        all_entities = json.loads(entity_data) if isinstance(entity_data, str) else entity_data
                        
                        # Search entities by name, type, or description
                        for entity_id, entity_info in all_entities.items():
                            name = entity_info.get("name", "").lower()
                            entity_type = entity_info.get("type", "").lower()
                            description = entity_info.get("description", "").lower()
                            
                            # Simple text matching
                            if (query_lower in name or 
                                query_lower in entity_type or 
                                query_lower in description or
                                any(query_lower in str(v).lower() for v in entity_info.values() if isinstance(v, str))):
                                
                                # Calculate relevance score (simple text matching)
                                score = 0.0
                                if query_lower in name:
                                    score += 1.0
                                if query_lower in entity_type:
                                    score += 0.8
                                if query_lower in description:
                                    score += 0.6
                                
                                entities.append({
                                    "id": entity_id,
                                    "name": entity_info.get("name", entity_id),
                                    "type": entity_info.get("type", "entity"),
                                    "description": entity_info.get("description", ""),
                                    "properties": entity_info,
                                    "relevance_score": score
                                })
                        
                        # Sort by relevance score and limit results
                        entities.sort(key=lambda x: x["relevance_score"], reverse=True)
                        entities = entities[:limit]
                        
                except Exception as e:
                    logger.warning(f"Error searching entities: {e}")
            
            return {
                "entities": entities,
                "query": query,
                "total_found": len(entities),
                "limit": limit,
                "search_metadata": {
                    "search_time": time.time(),
                    "search_type": "text_matching"
                }
            }
            
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            raise MCPError("SEARCH_FAILED", f"Entity search failed: {e}")

    async def update_entity(
        self, entity_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update entity in the knowledge graph."""
        await self._ensure_initialized()
        
        try:
            logger.info(f"Updating entity {entity_id}")
            
            # Access LightRAG's graph storage to update entity
            if hasattr(self._lightrag, "kg_storage"):
                try:
                    # Get current entities
                    entity_data = await self._lightrag.kg_storage.get("entities")
                    if entity_data:
                        all_entities = json.loads(entity_data) if isinstance(entity_data, str) else entity_data
                        
                        if entity_id not in all_entities:
                            raise MCPError("ENTITY_NOT_FOUND", f"Entity {entity_id} not found")
                        
                        # Update entity with new data
                        original_entity = all_entities[entity_id].copy()
                        all_entities[entity_id].update(updates)
                        
                        # Save updated entities back to storage
                        await self._lightrag.kg_storage.set("entities", json.dumps(all_entities))
                        
                        return {
                            "entity_id": entity_id,
                            "status": "updated",
                            "original": original_entity,
                            "updated": all_entities[entity_id],
                            "changes": updates,
                            "message": "Entity updated successfully"
                        }
                    else:
                        raise MCPError("NO_ENTITIES", "No entities found in storage")
                        
                except MCPError:
                    raise
                except Exception as e:
                    logger.error(f"Error updating entity: {e}")
                    raise MCPError("UPDATE_FAILED", f"Entity update failed: {e}")
            else:
                raise MCPError("STORAGE_UNAVAILABLE", "Graph storage not available")
                
        except MCPError:
            raise
        except Exception as e:
            logger.error(f"Entity update failed: {e}")
            raise MCPError("UPDATE_FAILED", f"Entity update failed: {e}")

    async def get_entity_relationships(
        self, entity_id: str, **params
    ) -> Dict[str, Any]:
        """Get relationships for a specific entity."""
        await self._ensure_initialized()
        
        try:
            logger.info(f"Getting relationships for entity {entity_id}")
            
            relationships = []
            limit = params.get("limit", 50)
            
            # Access LightRAG's graph storage to find relationships
            if hasattr(self._lightrag, "kg_storage"):
                try:
                    # First verify entity exists
                    entity_data = await self._lightrag.kg_storage.get("entities")
                    if entity_data:
                        all_entities = json.loads(entity_data) if isinstance(entity_data, str) else entity_data
                        if entity_id not in all_entities:
                            raise MCPError("ENTITY_NOT_FOUND", f"Entity {entity_id} not found")
                        
                        entity_info = all_entities[entity_id]
                    else:
                        raise MCPError("NO_ENTITIES", "No entities found in storage")
                    
                    # Get all relationships
                    relation_data = await self._lightrag.kg_storage.get("relationships")
                    if relation_data:
                        all_relationships = json.loads(relation_data) if isinstance(relation_data, str) else relation_data
                        
                        # Find relationships where this entity is source or target
                        for rel_id, rel_info in all_relationships.items():
                            source = rel_info.get("source", "")
                            target = rel_info.get("target", "")
                            
                            if source == entity_id or target == entity_id:
                                # Determine direction and connected entity
                                if source == entity_id:
                                    direction = "outgoing"
                                    connected_entity_id = target
                                else:
                                    direction = "incoming"
                                    connected_entity_id = source
                                
                                # Get connected entity info
                                connected_entity = all_entities.get(connected_entity_id, {})
                                
                                relationships.append({
                                    "id": rel_id,
                                    "type": rel_info.get("type", ""),
                                    "description": rel_info.get("description", ""),
                                    "direction": direction,
                                    "connected_entity": {
                                        "id": connected_entity_id,
                                        "name": connected_entity.get("name", connected_entity_id),
                                        "type": connected_entity.get("type", "entity")
                                    },
                                    "properties": rel_info
                                })
                        
                        # Limit results
                        relationships = relationships[:limit]
                        
                except Exception as e:
                    logger.warning(f"Error getting entity relationships: {e}")
            
            return {
                "entity_id": entity_id,
                "entity_name": entity_info.get("name", entity_id),
                "relationships": relationships,
                "total_found": len(relationships),
                "limit": limit,
                "relationship_summary": {
                    "incoming": len([r for r in relationships if r["direction"] == "incoming"]),
                    "outgoing": len([r for r in relationships if r["direction"] == "outgoing"]),
                    "total": len(relationships)
                }
            }
            
        except MCPError:
            raise
        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            raise MCPError("RELATIONSHIP_RETRIEVAL_FAILED", f"Entity relationship retrieval failed: {e}")

    # System operations
    async def health_check(self) -> Dict[str, Any]:
        """Check system health with comprehensive statistics."""
        try:
            await self._ensure_initialized()

            # Gather statistics
            total_documents = len(self._document_tracker)
            total_entities = 0
            total_relationships = 0
            
            if hasattr(self._lightrag, "kg_storage"):
                try:
                    # Count entities
                    entity_data = await self._lightrag.kg_storage.get("entities")
                    if entity_data:
                        entities = json.loads(entity_data) if isinstance(entity_data, str) else entity_data
                        total_entities = len(entities)
                    
                    # Count relationships
                    relation_data = await self._lightrag.kg_storage.get("relationships")
                    if relation_data:
                        relationships = json.loads(relation_data) if isinstance(relation_data, str) else relation_data
                        total_relationships = len(relationships)
                        
                except Exception as e:
                    logger.warning(f"Error gathering storage statistics: {e}")

            return {
                "status": "healthy",
                "version": "direct-mode",
                "uptime": "unknown",
                "configuration": {
                    "mode": "direct",
                    "working_dir": self.config.lightrag_working_dir or "./rag_storage",
                    "max_file_size_mb": self.config.max_file_size_mb,
                    "features": {
                        "streaming": True,
                        "document_tracking": True,
                        "graph_access": True,
                        "entity_search": True,
                        "entity_updates": True
                    }
                },
                "statistics": {
                    "total_documents": total_documents,
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "total_queries": self._query_stats["total_queries"],
                    "avg_response_time": (
                        self._query_stats["total_response_time"] / max(self._query_stats["total_queries"], 1)
                    ),
                },
                "query_stats": self._query_stats["queries_by_mode"],
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "LightRAG direct client not available",
            }

    async def get_system_stats(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get system statistics with available data."""
        try:
            # Calculate average response time
            avg_response_time = (
                self._query_stats["total_response_time"] / max(self._query_stats["total_queries"], 1)
            )
            
            return {
                "time_range": time_range,
                "query_statistics": {
                    "total_queries": self._query_stats["total_queries"],
                    "queries_by_mode": self._query_stats["queries_by_mode"],
                    "average_response_time": avg_response_time,
                    "cache_hit_rate": 0.0,  # Not implemented in direct mode
                },
                "document_statistics": {
                    "total_documents": len(self._document_tracker),
                    "documents_by_status": self._get_document_status_counts(),
                    "documents_by_type": self._get_document_type_counts(),
                },
                "storage_statistics": await self._get_storage_stats(),
                "note": "Statistics limited to current session in direct mode"
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                "time_range": time_range,
                "error": str(e),
                "message": "Failed to gather system statistics"
            }
    
    def _get_document_status_counts(self) -> Dict[str, int]:
        """Get count of documents by status."""
        status_counts = {}
        for doc in self._document_tracker.values():
            status = doc.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts
    
    def _get_document_type_counts(self) -> Dict[str, int]:
        """Get count of documents by type."""
        type_counts = {}
        for doc in self._document_tracker.values():
            doc_type = doc.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts
    
    async def _get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from LightRAG."""
        stats = {
            "entities": 0,
            "relationships": 0,
            "storage_type": "unknown"
        }
        
        if hasattr(self._lightrag, "kg_storage"):
            try:
                # Count entities
                entity_data = await self._lightrag.kg_storage.get("entities")
                if entity_data:
                    entities = json.loads(entity_data) if isinstance(entity_data, str) else entity_data
                    stats["entities"] = len(entities)
                
                # Count relationships
                relation_data = await self._lightrag.kg_storage.get("relationships")
                if relation_data:
                    relationships = json.loads(relation_data) if isinstance(relation_data, str) else relation_data
                    stats["relationships"] = len(relationships)
                
                # Get storage type
                stats["storage_type"] = type(self._lightrag.kg_storage).__name__
                
            except Exception as e:
                logger.warning(f"Error getting storage stats: {e}")
        
        return stats

    async def clear_cache(self, cache_types: List[str]) -> Dict[str, Any]:
        """Clear caches and reset statistics."""
        try:
            cleared = []
            
            for cache_type in cache_types:
                if cache_type == "query_stats":
                    # Reset query statistics
                    self._query_stats = {
                        "total_queries": 0,
                        "queries_by_mode": {},
                        "total_response_time": 0.0,
                    }
                    cleared.append("query_stats")
                
                elif cache_type == "document_tracking":
                    # Clear document tracking (careful operation)
                    self._document_tracker.clear()
                    cleared.append("document_tracking")
                
                elif cache_type == "lightrag_cache":
                    # Try to clear LightRAG's internal caches if available
                    if hasattr(self._lightrag, "clear_cache"):
                        await self._lightrag.clear_cache()
                        cleared.append("lightrag_cache")
                    else:
                        logger.warning("LightRAG cache clearing not available")
                
                else:
                    logger.warning(f"Unknown cache type: {cache_type}")
            
            return {
                "cleared_caches": cleared,
                "message": f"Cleared {len(cleared)} cache types",
                "available_cache_types": ["query_stats", "document_tracking", "lightrag_cache"],
                "note": "Cache clearing in direct mode has limited functionality"
            }
            
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            return {
                "cleared_caches": [],
                "error": str(e),
                "message": "Cache clearing failed"
            }
