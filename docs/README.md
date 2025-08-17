# LightRAG Documentation

Welcome to the comprehensive documentation for LightRAG, a production-ready Retrieval-Augmented Generation platform. This documentation is organized by user type and use case to help you quickly find the information you need.

## 📋 Documentation Overview

This documentation covers everything from basic usage to advanced deployment and customization. Choose your path based on your role and objectives:

### 👥 By User Type

#### 🚀 **New Users**
- Start with [Quick Start](#quick-start)
- Follow [Basic Tutorial](#tutorials--examples)
- Review [Configuration Guide](#configuration)

#### 👨‍💻 **Developers**
- Read [Developer Guide](development/README.md)
- Explore [API Documentation](api/README.md)
- Study [Architecture Overview](architecture/README.md)

#### 🔧 **DevOps/Operations**
- Review [Deployment Guide](deployment/README.md)
- Implement [Security Best Practices](#security)
- Setup [Monitoring & Observability](#monitoring)

#### 🏢 **Enterprise Users**
- Study [Production Deployment](#production-deployment)
- Review [Security & Compliance](#security)
- Plan [Scaling & Performance](#performance)

## 📚 Core Documentation

### 🎯 Quick Start
**Get up and running in 5 minutes**

- **[Installation](user-guide/installation.md)** - System requirements and installation
- **[First Steps](user-guide/quickstart.md)** - Your first document processing
- **[Basic Configuration](user-guide/configuration.md)** - Essential settings

### 🏗️ Architecture & Design
**Understand how LightRAG works**

- **[System Architecture](architecture/README.md)** - High-level system design
- **[Component Overview](architecture/components.md)** - Individual component details
- **[Data Flow](architecture/data-flow.md)** - How data moves through the system
- **[Storage Backends](architecture/storage.md)** - Storage layer architecture

### 🔌 API Reference
**Complete API documentation**

- **[REST API Guide](api/README.md)** - Complete REST API reference
- **[Authentication](api/authentication.md)** - JWT and API key authentication
- **[Error Handling](api/errors.md)** - Error codes and troubleshooting
- **[Rate Limiting](api/rate-limiting.md)** - Rate limiting and quotas

### 🚀 Deployment
**Production-ready deployment strategies**

- **[Deployment Guide](deployment/README.md)** - Comprehensive deployment options
- **[Docker Deployment](deployment/docker.md)** - Docker Compose setup
- **[Kubernetes Deployment](deployment/kubernetes.md)** - K8s deployment strategies
- **[Cloud Deployment](deployment/cloud.md)** - AWS, GCP, Azure deployment

### 👨‍💻 Development
**For developers and contributors**

- **[Developer Guide](development/README.md)** - Complete development guide
- **[Contributing](development/contributing.md)** - How to contribute
- **[Testing](development/testing.md)** - Testing strategies and best practices
- **[Debugging](development/debugging.md)** - Common issues and solutions

## 📖 User Guides

### 🌟 Getting Started
- **[Installation Guide](user-guide/installation.md)** - System setup and prerequisites
- **[Quick Start Tutorial](user-guide/quickstart.md)** - 5-minute getting started guide
- **[Configuration Overview](user-guide/configuration.md)** - Environment and settings
- **[First Document Processing](user-guide/first-document.md)** - Process your first document

### 🔧 Configuration
- **[Environment Variables](user-guide/environment.md)** - Complete environment reference
- **[LLM Provider Setup](user-guide/llm-providers.md)** - Configure OpenAI, Ollama, xAI, etc.
- **[xAI Integration Guide](user-guide/xai-integration.md)** - Complete xAI Grok model integration
- **[Ollama Integration Guide](user-guide/ollama-integration.md)** - Local embedding deployment and security
- **[PostgreSQL Integration](user-guide/postgresql-integration.md)** - Production database setup and security
- **[Storage Configuration](user-guide/storage.md)** - Choose and configure storage backends
- **[Security Settings](user-guide/security.md)** - Authentication and security configuration

### 📊 Usage Patterns
- **[Document Management](user-guide/documents.md)** - Upload, process, and manage documents
- **[Query Strategies](user-guide/querying.md)** - Different query modes and strategies
- **[Knowledge Graph](user-guide/knowledge-graph.md)** - Working with the knowledge graph
- **[Batch Processing](user-guide/batch-processing.md)** - Process multiple documents

## 🎓 Tutorials & Examples

### 📝 Basic Tutorials
- **[Basic Usage Tutorial](tutorials/basic-usage.md)** - Step-by-step introduction
- **[Document Processing Workflow](tutorials/document-workflow.md)** - Complete processing pipeline
- **[Query Optimization](tutorials/query-optimization.md)** - Optimize query performance
- **[Knowledge Graph Exploration](tutorials/knowledge-graph.md)** - Explore graph relationships

### 🔗 Integration Tutorials
- **[Python SDK Tutorial](tutorials/python-sdk.md)** - Using the Python library
- **[REST API Tutorial](tutorials/rest-api.md)** - Working with the REST API
- **[xAI Grok Integration](user-guide/xai-integration.md)** - Complete xAI integration guide
- **[Ollama Local Deployment](user-guide/ollama-integration.md)** - Privacy-focused local embeddings
- **[PostgreSQL Setup](user-guide/postgresql-integration.md)** - Production database configuration
- **[Claude MCP Integration](tutorials/claude-mcp.md)** - Model Context Protocol setup
- **[Web UI Tutorial](tutorials/web-ui.md)** - Using the React frontend

### 🏢 Advanced Use Cases
- **[Enterprise Deployment](tutorials/enterprise-deployment.md)** - Large-scale deployment
- **[Multi-Tenant Setup](tutorials/multi-tenant.md)** - Supporting multiple tenants
- **[Custom Storage Backends](tutorials/custom-storage.md)** - Implement custom storage
- **[Custom LLM Providers](tutorials/custom-llm.md)** - Add new LLM providers

## 🔒 Security

### 🛡️ Security Guides
- **[Security Overview](security/README.md)** - Comprehensive security guide
- **[Authentication & Authorization](security/auth.md)** - User authentication and access control
- **[PostgreSQL Security](user-guide/postgresql-integration.md#security-hardening)** - Database security hardening
- **[Ollama Security](user-guide/ollama-integration.md#security-hardening)** - Local deployment security
- **[Container Security](security/containers.md)** - Docker and Kubernetes security
- **[Network Security](security/network.md)** - Network configuration and firewall rules
- **[Data Protection](security/data-protection.md)** - Encrypt data at rest and in transit

### 🔐 Compliance
- **[Security Hardening](security/hardening.md)** - Production security checklist
- **[Audit Logging](security/audit-logging.md)** - Complete audit trail setup
- **[Vulnerability Management](security/vulnerabilities.md)** - Security scanning and updates
- **[Incident Response](security/incident-response.md)** - Security incident procedures

## 📊 Monitoring & Operations

### 📈 Monitoring
- **[Health Monitoring](monitoring/health-checks.md)** - System health and dependencies
- **[Performance Metrics](monitoring/metrics.md)** - Application and system metrics
- **[Logging](monitoring/logging.md)** - Structured logging and log aggregation
- **[Alerting](monitoring/alerting.md)** - Alert configuration and escalation

### 🔧 Operations
- **[Maintenance](operations/maintenance.md)** - Regular maintenance procedures
- **[Backup & Recovery](operations/backup.md)** - Data backup and disaster recovery
- **[Scaling](operations/scaling.md)** - Horizontal and vertical scaling
- **[Troubleshooting](operations/troubleshooting.md)** - Common issues and solutions

## 🎯 Reference

### 📋 Configuration Reference
- **[Environment Variables](reference/environment-variables.md)** - Complete environment reference
- **[Configuration Files](reference/configuration-files.md)** - Configuration file formats
- **[Storage Backends](reference/storage-backends.md)** - Available storage options
- **[LLM Providers](reference/llm-providers.md)** - Supported LLM providers

### 🔧 Technical Reference
- **[API Reference](reference/api-reference.md)** - Complete API specification
- **[CLI Reference](reference/cli-reference.md)** - Command-line interface
- **[Python API Reference](reference/python-api.md)** - Python library reference
- **[Error Codes](reference/error-codes.md)** - Error codes and solutions

### 📊 Performance Reference
- **[Benchmarks](reference/benchmarks.md)** - Performance benchmarks
- **[Capacity Planning](reference/capacity-planning.md)** - Resource requirements
- **[Optimization Guide](reference/optimization.md)** - Performance optimization
- **[Limits & Quotas](reference/limits.md)** - System limits and quotas

## 🔍 Search & Navigation

### Quick Links
- **[FAQ](faq.md)** - Frequently asked questions
- **[Glossary](glossary.md)** - Terms and definitions
- **[Changelog](../CHANGELOG.md)** - Version history and changes
- **[Migration Guide](migration/README.md)** - Upgrade and migration procedures

### Navigation Tips
- Use the search function to find specific topics
- Each guide includes cross-references to related documentation
- Code examples are provided throughout the documentation
- Screenshots and diagrams illustrate key concepts

## 🤝 Community & Support

### Getting Help
- **[Troubleshooting Guide](troubleshooting/README.md)** - Solve common problems
- **[GitHub Issues](https://github.com/your-repo/issues)** - Report bugs and request features
- **[Community Forum](https://community.lightrag.io)** - Ask questions and share experiences
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/lightrag)** - Technical Q&A

### Contributing
- **[Contributing Guide](development/contributing.md)** - How to contribute to LightRAG
- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community guidelines
- **[Development Setup](development/setup.md)** - Set up development environment
- **[Documentation Guidelines](development/documentation.md)** - Writing documentation

## 📱 Platform-Specific Guides

### Container Platforms
- **[Docker](platforms/docker.md)** - Docker deployment and configuration
- **[Kubernetes](platforms/kubernetes.md)** - Kubernetes deployment strategies
- **[Docker Swarm](platforms/docker-swarm.md)** - Docker Swarm orchestration

### Cloud Platforms
- **[AWS](platforms/aws.md)** - Amazon Web Services deployment
- **[Google Cloud](platforms/gcp.md)** - Google Cloud Platform deployment
- **[Azure](platforms/azure.md)** - Microsoft Azure deployment
- **[DigitalOcean](platforms/digitalocean.md)** - DigitalOcean deployment

### Local Development
- **[Local Setup](platforms/local.md)** - Local development environment
- **[IDE Configuration](platforms/ide.md)** - IDE setup and configuration
- **[Debugging](platforms/debugging.md)** - Local debugging techniques

## 📅 Version-Specific Documentation

### Current Version (v1.5.0)
- **[Release Notes](releases/v1.5.0.md)** - What's new in v1.5.0
- **[Migration from v1.4.x](migration/v1.4-to-v1.5.md)** - Upgrade procedures
- **[Breaking Changes](migration/breaking-changes.md)** - Important changes

### Previous Versions
- **[v1.4.x Documentation](archive/v1.4/)** - Archived documentation
- **[v1.3.x Documentation](archive/v1.3/)** - Archived documentation
- **[Version History](releases/history.md)** - Complete version history

## 📋 Documentation Maintenance

This documentation is actively maintained and updated. If you find any issues or have suggestions for improvement:

1. **Report Issues**: Create an issue on GitHub for documentation bugs
2. **Suggest Improvements**: Submit pull requests for documentation updates
3. **Request Topics**: Ask for additional documentation topics in GitHub discussions

**Last Updated**: 2025-01-17
**Documentation Version**: 1.5.0
**Target Audience**: All LightRAG users and developers

---

## Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| **Get started quickly** | [Quick Start](user-guide/quickstart.md) |
| **Deploy to production** | [Deployment Guide](deployment/README.md) |
| **Develop with LightRAG** | [Developer Guide](development/README.md) |
| **Use the API** | [API Documentation](api/README.md) |
| **Understand the architecture** | [Architecture Guide](architecture/README.md) |
| **Secure my deployment** | [Security Guide](security/README.md) |
| **Monitor and operate** | [Operations Guide](operations/README.md) |
| **Troubleshoot issues** | [Troubleshooting Guide](troubleshooting/README.md) |

**Need immediate help?** Check the [FAQ](faq.md) or [Troubleshooting Guide](troubleshooting/README.md).