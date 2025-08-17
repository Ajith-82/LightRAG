# LightRAG API Documentation

## Overview

LightRAG provides a comprehensive REST API for document processing, knowledge graph operations, and intelligent querying. The API is built on FastAPI and provides both synchronous and asynchronous operations.

## Base URL

- **Development**: `http://localhost:9621`
- **Production**: `https://your-domain.com`

## Authentication

LightRAG supports JWT-based authentication:

```bash
# Get access token
curl -X POST "http://localhost:9621/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" "http://localhost:9621/api/health"
```

## Core Endpoints

### Health & Status

#### GET `/health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

#### GET `/api/health`
Detailed health check with system information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.5.0",
  "storage": {
    "kv": "connected",
    "vector": "connected",
    "graph": "connected"
  },
  "llm": "connected"
}
```

### Document Operations

#### POST `/documents/upload`
Upload and process documents.

**Request:**
```bash
curl -X POST "http://localhost:9621/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "document_id": "doc_123",
  "status": "processing",
  "message": "Document uploaded successfully"
}
```

#### POST `/documents/text`
Insert text content directly.

**Request:**
```json
{
  "content": "Your text content here",
  "metadata": {
    "source": "manual_input",
    "title": "Sample Document"
  }
}
```

**Response:**
```json
{
  "document_id": "doc_124",
  "status": "completed",
  "chunks_created": 5
}
```

#### GET `/documents`
List all documents.

**Parameters:**
- `limit` (optional): Number of documents to return (default: 50)
- `offset` (optional): Number of documents to skip (default: 0)

**Response:**
```json
{
  "documents": [
    {
      "id": "doc_123",
      "title": "Sample Document",
      "status": "completed",
      "created_at": "2025-01-15T10:30:00Z",
      "chunk_count": 5
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### GET `/documents/{document_id}`
Get document details.

**Response:**
```json
{
  "id": "doc_123",
  "title": "Sample Document",
  "status": "completed",
  "created_at": "2025-01-15T10:30:00Z",
  "metadata": {
    "source": "upload",
    "file_type": "pdf"
  },
  "chunks": [
    {
      "id": "chunk_1",
      "content": "Chunk content...",
      "position": 0
    }
  ]
}
```

#### DELETE `/documents/{document_id}`
Delete a document and its associated data.

**Response:**
```json
{
  "message": "Document deleted successfully",
  "document_id": "doc_123"
}
```

### Query Operations

#### POST `/query`
Perform RAG queries with different modes.

**Request:**
```json
{
  "query": "What are the main concepts in machine learning?",
  "mode": "hybrid",
  "top_k": 10,
  "include_context": true
}
```

**Query Modes:**
- `local`: Context-dependent information retrieval
- `global`: Global knowledge graph queries  
- `hybrid`: Combines local and global methods
- `mix`: Integrates knowledge graph and vector retrieval
- `naive`: Basic vector search without graph enhancement

**Response:**
```json
{
  "answer": "Machine learning encompasses several main concepts...",
  "sources": [
    {
      "document_id": "doc_123",
      "chunk_id": "chunk_1",
      "relevance_score": 0.95,
      "content": "Source content..."
    }
  ],
  "mode": "hybrid",
  "processing_time": 1.234
}
```

### Knowledge Graph Operations

#### GET `/graph/entities`
List entities in the knowledge graph.

**Parameters:**
- `limit` (optional): Number of entities to return
- `search` (optional): Search term for entity names

**Response:**
```json
{
  "entities": [
    {
      "id": "entity_1",
      "name": "Machine Learning",
      "type": "concept",
      "connections": 15
    }
  ],
  "total": 1
}
```

#### GET `/graph/relationships`
List relationships in the knowledge graph.

**Response:**
```json
{
  "relationships": [
    {
      "source": "Machine Learning",
      "target": "Neural Networks",
      "type": "includes",
      "weight": 0.8
    }
  ],
  "total": 1
}
```

#### GET `/graph/stats`
Get knowledge graph statistics.

**Response:**
```json
{
  "entities": 1250,
  "relationships": 3400,
  "entity_types": {
    "concept": 800,
    "person": 200,
    "organization": 150,
    "other": 100
  }
}
```

### Chat Interface (Ollama Compatible)

#### POST `/api/chat`
Ollama-compatible chat interface.

**Request:**
```json
{
  "model": "lightrag",
  "messages": [
    {
      "role": "user", 
      "content": "What is machine learning?"
    }
  ],
  "stream": false
}
```

**Response:**
```json
{
  "model": "lightrag",
  "created_at": "2025-01-15T10:30:00Z",
  "message": {
    "role": "assistant",
    "content": "Machine learning is a subset of artificial intelligence..."
  },
  "done": true
}
```

## Error Handling

All endpoints return consistent error responses:

**400 Bad Request:**
```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": {
    "field": "query",
    "issue": "Field is required"
  }
}
```

**401 Unauthorized:**
```json
{
  "error": "unauthorized",
  "message": "Invalid or expired token"
}
```

**429 Too Many Requests:**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 60
}
```

**500 Internal Server Error:**
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "request_id": "req_123"
}
```

## Rate Limiting

API endpoints are rate-limited based on authentication:

- **Authenticated users**: 1000 requests/hour
- **Unauthenticated**: 100 requests/hour
- **Upload endpoints**: 50 requests/hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642612800
```

## WebSocket Support

Real-time updates for document processing:

```javascript
const ws = new WebSocket('ws://localhost:9621/ws/documents');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Document status:', update);
};
```

## SDKs and Examples

### Python SDK
```python
from lightrag_client import LightRAGClient

client = LightRAGClient("http://localhost:9621", token="your_token")

# Upload document
result = client.upload_document("document.pdf")

# Query
response = client.query("What is machine learning?", mode="hybrid")
```

### JavaScript/Node.js
```javascript
const LightRAG = require('lightrag-js');

const client = new LightRAG({
  baseURL: 'http://localhost:9621',
  token: 'your_token'
});

// Query
const response = await client.query({
  query: 'What is machine learning?',
  mode: 'hybrid'
});
```

### cURL Examples

**Upload and query workflow:**
```bash
# 1. Upload document
curl -X POST "http://localhost:9621/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@research_paper.pdf"

# 2. Wait for processing (check status)
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:9621/documents/doc_123"

# 3. Query the knowledge base
curl -X POST "http://localhost:9621/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings?",
    "mode": "hybrid",
    "top_k": 5
  }'
```

## OpenAPI Specification

The complete OpenAPI specification is available at:
- **JSON**: `http://localhost:9621/openapi.json`
- **Interactive docs**: `http://localhost:9621/docs`
- **ReDoc**: `http://localhost:9621/redoc`

## Configuration

API behavior can be configured via environment variables:

```bash
# Server configuration
PORT=9621
HOST=0.0.0.0
WORKERS=4

# Authentication
AUTH_ENABLED=true
JWT_SECRET_KEY=your_secret_key
JWT_EXPIRE_HOURS=24

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# LLM configuration
LLM_BINDING=openai
LLM_MODEL=gpt-4
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
```

## Monitoring

### Health Monitoring
Regular health checks should be performed:
```bash
# Basic health check
curl http://localhost:9621/health

# Detailed health with dependencies
curl http://localhost:9621/api/health
```

### Metrics
Application metrics are available at:
- Response times
- Request counts
- Error rates
- Active connections

### Logging
All API requests are logged with:
- Request timestamp
- User identification
- Endpoint accessed
- Response status
- Processing time

See the deployment documentation for monitoring setup details.