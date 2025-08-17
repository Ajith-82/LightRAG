# LightRAG Architecture Documentation

## System Overview

LightRAG is a production-ready Retrieval-Augmented Generation (RAG) system that combines knowledge graphs with vector retrieval for enhanced document processing and intelligent querying. The architecture is designed for scalability, reliability, and enterprise deployment.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   API Clients   │    │   MCP Client    │
│  (React/TS)     │    │   (Python/JS)   │    │   (Claude)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      LightRAG API        │
                    │    (FastAPI Server)      │
                    └─────────────┬─────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
    ┌──────────▼────────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │   Storage Layer   │ │  LLM Layer  │ │ Processing  │
    │  (Multi-Backend)  │ │ (Multiple   │ │  Pipeline   │
    │                   │ │  Providers) │ │             │
    └───────────────────┘ └─────────────┘ └─────────────┘
```

## Core Components

### 1. API Server Layer (`lightrag/api/`)

**FastAPI-based REST API** providing:
- Document upload and processing endpoints
- RAG query interface with multiple modes
- Knowledge graph exploration APIs
- Ollama-compatible chat interface
- Authentication and authorization
- Rate limiting and audit logging

**Key Files:**
- `lightrag/api/app.py` - Main FastAPI application
- `lightrag/api/routers/` - API endpoint definitions
- `lightrag/api/auth/` - Authentication system
- `lightrag/api/middleware/` - Security middleware

### 2. Core Processing Engine (`lightrag/`)

**Document Processing Pipeline:**
```
Document Input → Text Extraction → Chunking → Embedding → Knowledge Graph → Storage
```

**Key Components:**
- **LightRAG Core** (`lightrag.py`): Main orchestrator class
- **Operation Handler** (`operate.py`): Document processing operations
- **Knowledge Graph Builder**: Entity and relationship extraction
- **Embedding Pipeline**: Vector representation generation

### 3. Storage Architecture (`lightrag/kg/`)

LightRAG uses a **4-layer storage architecture** with pluggable backends:

#### Storage Types:

1. **KV Storage** - Document chunks and LLM cache
   - `JsonKVStorage` - File-based storage
   - `PGKVStorage` - PostgreSQL backend
   - `RedisKVStorage` - Redis backend
   - `MongoKVStorage` - MongoDB backend

2. **Vector Storage** - Embedding vectors and similarity search
   - `NanoVectorDBStorage` - Lightweight vector database
   - `PGVectorStorage` - PostgreSQL with pgvector
   - `MilvusVectorDBStorage` - Milvus vector database
   - `QdrantStorage` - Qdrant vector database

3. **Graph Storage** - Entity relationships and graph operations
   - `NetworkXStorage` - In-memory graph (development)
   - `Neo4JStorage` - Neo4j graph database
   - `PGGraphStorage` - PostgreSQL with graph extensions
   - `MemgraphStorage` - Memgraph database

4. **Document Status Storage** - Processing status tracking
   - `JsonDocStatusStorage` - File-based status
   - `PGDocStatusStorage` - PostgreSQL status tracking
   - `MongoDocStatusStorage` - MongoDB status tracking

#### Storage Selection Matrix:

| Use Case | KV Storage | Vector Storage | Graph Storage | Document Status |
|----------|------------|----------------|---------------|-----------------|
| **Development** | JsonKV | NanoVectorDB | NetworkX | JsonDocStatus |
| **Small Production** | Redis | PGVector | PGGraph | PGDocStatus |
| **Large Scale** | MongoDB | Milvus/Qdrant | Neo4j | MongoDocStatus |
| **Enterprise** | PostgreSQL | PGVector | Neo4j | PostgreSQL |

### 4. LLM Integration Layer (`lightrag/llm/`)

**Multi-Provider Support:**
- **OpenAI** - GPT models and embeddings
- **Ollama** - Local LLM deployment
- **Azure OpenAI** - Enterprise OpenAI service
- **xAI** - Grok models with specialized handling
- **Anthropic** - Claude models (via MCP)

**Provider Interface:**
```python
class BaseLLM:
    async def agenerate(self, messages: List[Message]) -> str
    async def aembedding(self, texts: List[str]) -> List[List[float]]
```

### 5. Web UI (`lightrag_webui/`)

**React/TypeScript Frontend:**
- Document upload and management interface
- Knowledge graph visualization
- Query interface with real-time results
- System monitoring and health dashboard

**Technology Stack:**
- React 18 with TypeScript
- Vite build system
- Bun package manager (optional)
- Modern CSS with responsive design

### 6. MCP Server (`lightrag_mcp/`)

**Model Context Protocol Integration:**
- 11 tools for document and graph operations
- 3 resources for system information
- Streaming support for real-time responses
- Direct and API-based operation modes

## Data Flow Architecture

### Document Processing Flow

```
┌─────────────┐
│   Upload    │
│  Document   │
└──────┬──────┘
       │
┌──────▼──────┐
│   Extract   │
│    Text     │
└──────┬──────┘
       │
┌──────▼──────┐
│   Chunk     │
│  Content    │
└──────┬──────┘
       │
┌──────▼──────┐
│  Generate   │
│ Embeddings  │
└──────┬──────┘
       │
┌──────▼──────┐
│   Extract   │
│  Entities   │
└──────┬──────┘
       │
┌──────▼──────┐
│    Build    │
│    Graph    │
└──────┬──────┘
       │
┌──────▼──────┐
│    Store    │
│    Data     │
└─────────────┘
```

### Query Processing Flow

```
┌─────────────┐
│    Query    │
│    Input    │
└──────┬──────┘
       │
┌──────▼──────┐
│   Analyze   │
│    Query    │
└──────┬──────┘
       │
┌──────▼──────┐    ┌─────────────┐
│   Vector    │    │    Graph    │
│   Search    │◄──►│   Search    │
└──────┬──────┘    └──────┬──────┘
       │                  │
       └──────┬──────┬────┘
              │      │
        ┌─────▼──────▼─────┐
        │    Combine &     │
        │     Rank         │
        └─────────┬────────┘
                  │
        ┌─────────▼────────┐
        │   Generate       │
        │   Response       │
        └──────────────────┘
```

## Security Architecture

### Authentication & Authorization

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │    │     API     │    │    Auth     │
│  Request    │───►│  Gateway    │───►│   Service   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                   ┌──────▼──────┐
                   │    Rate     │
                   │   Limiter   │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │   Audit     │
                   │   Logger    │
                   └─────────────┘
```

**Security Features:**
- JWT-based authentication with configurable expiration
- Role-based access control (RBAC)
- Rate limiting per user/endpoint
- Request/response audit logging
- Input validation and sanitization
- CORS and security headers

### Container Security

**Security Hardening:**
- Non-root user execution (UID >1000)
- Read-only root filesystem
- Minimal container capabilities
- Network segmentation
- Secret management via environment variables

## Deployment Architecture

### Development Environment

```
┌─────────────────────────────────────────────────────────┐
│                Docker Compose                           │
├─────────────┬─────────────┬─────────────┬─────────────┤
│  LightRAG   │ PostgreSQL  │    Redis    │   Web UI    │
│   Server    │             │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Production Environment

```
┌─────────────────────────────────────────────────────────┐
│                   Load Balancer                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────┐
│                 Kubernetes                              │
├─────────────┬────────┼────────┬─────────────┬─────────────┤
│  LightRAG   │ PostgreSQL     │    Redis    │   Web UI    │
│  Pods (3x)  │   Primary      │   Cluster   │    Pods     │
│             │  + Replica     │             │             │
└─────────────┴────────────────┴─────────────┴─────────────┘
```

### High Availability Setup

**Components:**
- **Load Balancer**: HAProxy/Nginx for request distribution
- **API Servers**: Multiple LightRAG instances
- **Database**: PostgreSQL with replication
- **Cache**: Redis cluster
- **Storage**: Persistent volumes with backups

## Performance Architecture

### Scaling Strategies

**Horizontal Scaling:**
- Multiple API server instances
- Database read replicas
- Redis clustering
- CDN for static assets

**Vertical Scaling:**
- Increased container resources
- Database performance tuning
- Optimized embeddings batch size

### Caching Strategy

**Multi-Level Caching:**
1. **LLM Response Cache** - Redis-based caching of LLM responses
2. **Embedding Cache** - Vector embedding caching
3. **Query Result Cache** - Frequently accessed query results
4. **Database Query Cache** - PostgreSQL query result caching

### Performance Monitoring

**Metrics Collection:**
- Response time percentiles
- Request throughput
- Error rates
- Resource utilization
- Database performance

## Configuration Architecture

### Environment-Based Configuration

```
production.env     →  Production settings
development.env    →  Development settings
testing.env        →  Test environment settings
```

### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env files)
3. **Default Values** (lowest priority)

### Key Configuration Areas

**LLM Configuration:**
```bash
LLM_BINDING=openai
LLM_MODEL=gpt-4
LLM_API_KEY=sk-...
MAX_ASYNC=5
TIMEOUT=120
```

**Storage Configuration:**
```bash
KV_STORAGE=postgres
VECTOR_STORAGE=pgvector
GRAPH_STORAGE=neo4j
DOC_STATUS_STORAGE=postgres
```

**Security Configuration:**
```bash
AUTH_ENABLED=true
JWT_SECRET_KEY=your-secret-key
JWT_EXPIRE_HOURS=24
RATE_LIMIT_ENABLED=true
```

## Monitoring & Observability

### Health Checks

**Multi-Level Health Monitoring:**
1. **Basic Health** - Application responsiveness
2. **Dependency Health** - Database/cache connectivity
3. **LLM Health** - Provider availability
4. **Storage Health** - Backend status

### Logging Architecture

**Structured Logging:**
- Application logs (JSON format)
- Access logs (request/response)
- Audit logs (security events)
- Error logs (exception tracking)

### Metrics & Alerting

**Key Metrics:**
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates by endpoint
- Resource utilization
- Queue depths

## Extension Points

### Custom Storage Backends

Implement the storage interfaces:
```python
class CustomKVStorage(BaseKVStorage):
    async def get(self, key: str) -> Any
    async def set(self, key: str, value: Any) -> None
    async def delete(self, key: str) -> None
```

### Custom LLM Providers

Implement the LLM interface:
```python
class CustomLLM(BaseLLM):
    async def agenerate(self, messages: List[Message]) -> str
    async def aembedding(self, texts: List[str]) -> List[List[float]]
```

### Custom Processing Pipeline

Extend the operation handler:
```python
class CustomOperator(BaseOperator):
    async def process_document(self, content: str) -> ProcessingResult
```

## Future Architecture Considerations

### Planned Enhancements

1. **Microservices Architecture** - Breaking down components
2. **Event-Driven Architecture** - Async processing with message queues
3. **Multi-Tenant Support** - Isolated tenant environments
4. **Advanced Security** - OAuth2, SAML integration
5. **ML Pipeline Integration** - Custom model training

### Scalability Roadmap

1. **Phase 1**: Container orchestration (current)
2. **Phase 2**: Microservices decomposition
3. **Phase 3**: Multi-region deployment
4. **Phase 4**: Edge computing integration

This architecture provides a solid foundation for enterprise-grade RAG applications while maintaining flexibility for future enhancements and scaling requirements.