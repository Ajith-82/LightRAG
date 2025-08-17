# LightRAG

**Production-Ready Retrieval-Augmented Generation Platform**

LightRAG is an enterprise-grade RAG (Retrieval-Augmented Generation) system that combines knowledge graphs with vector retrieval for enhanced document processing and intelligent querying. Built for scalability, security, and production deployment.

> **Based on**: This implementation is based on the original [LightRAG](https://github.com/HKUDS/LightRAG) research project by Hong Kong University's Data Science Lab, with significant production enhancements and enterprise features.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

### 🏗️ **Production-Ready Architecture**
- **Multi-Storage Backend Support**: PostgreSQL, Redis, MongoDB, Neo4j, Qdrant, Milvus
- **Scalable Deployment**: Docker Compose, Kubernetes, cloud platforms
- **Enterprise Security**: JWT authentication, rate limiting, audit logging
- **High Availability**: Load balancing, connection pooling, health monitoring

### 🤖 **Advanced AI Integration**
- **Multi-LLM Support**: OpenAI, Ollama, Azure OpenAI, xAI (Grok), Anthropic
- **Hybrid Retrieval**: Knowledge graph + vector search for superior accuracy
- **Multiple Query Modes**: Local, global, hybrid, mix, and naive search
- **Async Processing**: High-performance concurrent operations

### 🌐 **Modern Web Stack**
- **REST API**: FastAPI-based with OpenAPI documentation
- **Web UI**: React/TypeScript frontend with real-time updates
- **MCP Integration**: Model Context Protocol for Claude CLI
- **Ollama Compatible**: Drop-in replacement for Ollama chat API

### 🔒 **Enterprise Security**
- **Authentication**: JWT-based with configurable expiration
- **Authorization**: Role-based access control
- **Rate Limiting**: Configurable per-endpoint limits
- **Container Security**: Non-root users, read-only filesystems
- **Audit Logging**: Complete request/response audit trail

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- Python 3.10+ (for local development)
- API key for LLM provider (OpenAI, xAI, etc.)

### 1. Development Setup (5 minutes)

```bash
# Clone repository
git clone <repository-url>
cd LightRAG

# Configure environment
cp env.example .env
# Edit .env with your API keys

# Start services
docker compose up -d

# Verify deployment
curl http://localhost:9621/health
```

**Access Points:**
- **API Server**: http://localhost:9621
- **API Documentation**: http://localhost:9621/docs
- **Web UI**: http://localhost:3000

### 2. Production Deployment (15 minutes)

```bash
# Use production configuration
cp production.env .env
# Configure production settings in .env

# Deploy with security hardening
docker compose -f docker-compose.production.yml up -d

# Verify production deployment
curl http://localhost:9621/api/health
```

### 3. First Document Processing

```bash
# Upload a document
curl -X POST "http://localhost:9621/documents/upload" \
  -F "file=@your_document.pdf"

# Query the knowledge base
curl -X POST "http://localhost:9621/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main topics?",
    "mode": "hybrid"
  }'
```

## 📖 Documentation

### Getting Started
- **[Quick Start Guide](docs/user-guide/quickstart.md)** - Get up and running in minutes
- **[Installation Guide](docs/deployment/README.md)** - Comprehensive deployment options
- **[Configuration Guide](docs/user-guide/configuration.md)** - Environment and settings

### For Developers
- **[Developer Guide](docs/development/README.md)** - Development setup and workflow
- **[API Documentation](docs/api/README.md)** - Complete REST API reference
- **[Architecture Guide](docs/architecture/README.md)** - System design and components

### For Operations
- **[Deployment Guide](docs/deployment/README.md)** - Production deployment strategies
- **[Security Guide](docs/security/README.md)** - Security hardening and best practices
- **[Monitoring Guide](docs/monitoring/README.md)** - Health checks and observability

### Tutorials & Examples
- **[Basic Usage Tutorial](docs/tutorials/basic-usage.md)** - Step-by-step introduction
- **[Advanced Features](docs/tutorials/advanced-features.md)** - Knowledge graphs and custom processing
- **[Integration Examples](docs/tutorials/integrations.md)** - Connect with external systems

## 🏗️ Architecture Overview

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

### Core Components

- **LightRAG Core**: Document processing and knowledge graph construction
- **API Server**: FastAPI-based REST API with authentication and rate limiting
- **Storage Backends**: Pluggable storage with 8+ backend options
- **LLM Integration**: Multi-provider support with async processing
- **Web UI**: Modern React frontend for document management
- **MCP Server**: Model Context Protocol integration for Claude CLI

## 🔧 Configuration

### Environment Variables

```bash
# LLM Provider
LLM_BINDING=openai               # openai, ollama, azure_openai, xai
LLM_MODEL=gpt-4                  # Model name
OPENAI_API_KEY=sk-...           # API key

# Storage Backends
KV_STORAGE=postgres             # postgres, redis, mongo, json
VECTOR_STORAGE=pgvector         # pgvector, milvus, qdrant, nano
GRAPH_STORAGE=postgres          # postgres, neo4j, networkx

# Security
AUTH_ENABLED=true               # Enable authentication
JWT_SECRET_KEY=your-secret      # JWT signing key
RATE_LIMIT_ENABLED=true        # Enable rate limiting

# Performance
WORKERS=4                       # Gunicorn workers
MAX_ASYNC=5                     # Concurrent operations
```

### Storage Backend Selection

| Use Case | KV Storage | Vector Storage | Graph Storage |
|----------|------------|----------------|---------------|
| **Development** | JsonKV | NanoVectorDB | NetworkX |
| **Small Production** | Redis | PGVector | PGGraph |
| **Large Scale** | MongoDB | Milvus/Qdrant | Neo4j |
| **Enterprise** | PostgreSQL | PGVector | Neo4j |

## 🔌 Integration Examples

### Python SDK
```python
from lightrag import LightRAG

# Initialize LightRAG
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model="gpt-4",
    embedding_model="text-embedding-ada-002"
)

# Process document
await rag.ainsert("Your document content here")

# Query knowledge base
response = await rag.aquery(
    "What are the main concepts?",
    mode="hybrid"
)
print(response)
```

### REST API
```bash
# Upload document
curl -X POST "http://localhost:9621/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"

# Query with hybrid mode
curl -X POST "http://localhost:9621/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key findings",
    "mode": "hybrid",
    "top_k": 10
  }'
```

### Claude CLI (MCP)
```bash
# Setup MCP integration
claude config mcp add lightrag-mcp python -m lightrag_mcp

# Query through Claude
claude mcp lightrag_query "What are the main themes?" --mode hybrid

# Upload document
claude mcp lightrag_insert_file "/path/to/document.pdf"
```

## 🚀 Deployment Options

### Docker Compose (Recommended for Development)
```bash
# Development stack
docker compose up -d

# Production stack with security
docker compose -f docker-compose.production.yml up -d
```

### Kubernetes (Recommended for Production)
```bash
cd k8s-deploy
./install_lightrag.sh

# Monitor deployment
kubectl get pods -n lightrag
```

### Cloud Platforms
- **AWS**: ECS, EKS, Lambda
- **GCP**: Cloud Run, GKE
- **Azure**: Container Instances, AKS

## 🧪 Query Modes

LightRAG supports multiple query modes for different use cases:

- **`local`**: Context-dependent information retrieval
- **`global`**: Global knowledge graph queries
- **`hybrid`**: Combines local and global methods (recommended)
- **`mix`**: Integrates knowledge graph and vector retrieval
- **`naive`**: Basic vector search without graph enhancement

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Configurable per-endpoint rate limits
- **Audit Logging**: Complete request/response audit trail
- **Container Security**: Non-root users, minimal capabilities
- **Input Validation**: SQL injection and XSS prevention
- **Network Segmentation**: Internal container networks

## 📊 Performance

- **Async Architecture**: High-performance concurrent processing
- **Connection Pooling**: Optimized database connections
- **Caching**: Multi-level caching for LLM responses and embeddings
- **Batch Processing**: Efficient bulk operations
- **Resource Monitoring**: Built-in performance metrics

## 🛠️ Development

### Local Development Setup
```bash
# Clone and setup
git clone <repository-url>
cd LightRAG

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,test]"

# Run tests
pytest

# Start development server
lightrag-server
```

### Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lightrag

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
```

## 📄 License

LightRAG is released under the [MIT License](LICENSE).

## 🙏 Acknowledgments

This project builds upon the foundational research and implementation from:

- **Original LightRAG**: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) - The original research implementation by Hong Kong University's Data Science Lab
- **Research Paper**: ["LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://arxiv.org/abs/2410.05779) - The academic foundation for this work
- **Core Algorithms**: The knowledge graph construction and hybrid retrieval algorithms are based on the original research

This production-ready implementation extends the original work with enterprise features, security enhancements, multi-storage backends, and production deployment capabilities while maintaining compatibility with the core LightRAG algorithms.

## 🤝 Support

- **Documentation**: Comprehensive guides in the [docs/](docs/) directory
- **Issues**: Report bugs and request features on GitHub Issues
- **Community**: Join discussions and get help from the community

## 🗺️ Roadmap

### Current (v1.5.0)
- ✅ Production-ready deployment
- ✅ Multi-storage backend support
- ✅ Enterprise security features
- ✅ Comprehensive documentation

### Upcoming (v2.0.0)
- 🔄 Storage backend migration support
- 🔄 Advanced monitoring and alerting
- 🔄 Multi-tenant support
- 🔄 Performance optimizations

## 🎯 Use Cases

- **Enterprise Knowledge Management**: Centralized document processing and querying
- **Research and Analytics**: Academic and scientific document analysis
- **Customer Support**: Intelligent knowledge base for support teams
- **Content Management**: Large-scale content organization and retrieval
- **AI-Powered Applications**: Backend for AI-driven applications

---

**Ready to get started?** Follow our [Quick Start Guide](docs/user-guide/quickstart.md) or explore the [API Documentation](docs/api/README.md).

For production deployments, see our [Deployment Guide](docs/deployment/README.md) and [Security Best Practices](docs/security/README.md).