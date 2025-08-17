# LightRAG Deployment Guide

## Overview

This guide covers deployment options for LightRAG, from development setup to production-grade enterprise deployments. LightRAG supports multiple deployment strategies including Docker Compose, Kubernetes, and cloud platforms.

## Quick Start Deployment

### Prerequisites

**System Requirements:**
- Docker 20.10+ and Docker Compose 2.0+
- Python 3.10+ (for local development)
- Minimum 4GB RAM, 10GB storage
- Network access for LLM API calls

**Required API Keys:**
- OpenAI API key (or alternative LLM provider)
- Embedding model access (OpenAI, Ollama, etc.)

### Development Deployment (5 minutes)

1. **Clone and configure:**
```bash
git clone <repository-url>
cd LightRAG
cp env.example .env
```

2. **Configure environment (.env):**
```bash
# Essential settings
LLM_BINDING=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-your-api-key

EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
```

3. **Start services:**
```bash
docker compose up -d
```

4. **Verify deployment:**
```bash
curl http://localhost:9621/health
# Expected: {"status": "healthy"}
```

**Access Points:**
- API Server: http://localhost:9621
- API Documentation: http://localhost:9621/docs
- Web UI: http://localhost:3000 (if enabled)

### Production Deployment (15 minutes)

1. **Setup production environment:**
```bash
cp production.env .env
# Edit .env with production settings
```

2. **Configure production settings:**
```bash
# Security
AUTH_ENABLED=true
JWT_SECRET_KEY=your-secure-secret-key-min-32-chars
JWT_EXPIRE_HOURS=24

# Performance
WORKERS=4
MAX_ASYNC=5

# Database (recommended for production)
KV_STORAGE=postgres
VECTOR_STORAGE=pgvector
GRAPH_STORAGE=postgres
```

3. **Deploy with production configuration:**
```bash
docker compose -f docker-compose.production.yml up -d
```

4. **Verify production deployment:**
```bash
curl http://localhost:9621/api/health
```

## Deployment Options

### 1. Docker Compose Deployment

#### Standard Development Stack
```yaml
# docker-compose.yml
services:
  lightrag:
    build: .
    ports:
      - "9621:9621"
    environment:
      - LLM_BINDING=openai
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./rag_storage:/app/rag_storage
```

#### Enhanced PostgreSQL Stack
```bash
# Use enhanced PostgreSQL with pgvector and Apache AGE
docker compose -f docker-compose.enhanced.yml up -d

# Features:
# - PostgreSQL 16 with pgvector extension
# - Apache AGE graph database
# - Redis for caching
# - Performance optimization
```

#### Production Security Stack
```bash
# Security-hardened production deployment
docker compose -f docker-compose.production.yml up -d

# Features:
# - Non-root containers (UID >1000)
# - Read-only filesystems
# - Network segmentation
# - JWT authentication
# - Rate limiting
# - Audit logging
```

### 2. Kubernetes Deployment

#### Prerequisites
```bash
# Kubernetes cluster (1.20+)
# kubectl configured
# Helm 3.0+ (optional)
```

#### Quick Kubernetes Deployment
```bash
cd k8s-deploy

# 1. Install dependencies
./databases/01-prepare.sh
./databases/02-install-database.sh

# 2. Deploy LightRAG
./install_lightrag.sh

# 3. Verify deployment
kubectl get pods -n lightrag
kubectl logs -f deployment/lightrag -n lightrag
```

#### Manual Kubernetes Deployment
```yaml
# lightrag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag
  namespace: lightrag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lightrag
  template:
    metadata:
      labels:
        app: lightrag
    spec:
      containers:
      - name: lightrag
        image: lightrag:latest
        ports:
        - containerPort: 9621
        env:
        - name: LLM_BINDING
          value: "openai"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: lightrag-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

#### Helm Deployment
```bash
# Add LightRAG Helm repository
helm repo add lightrag https://charts.lightrag.io
helm repo update

# Install with custom values
helm install lightrag lightrag/lightrag \
  --namespace lightrag \
  --create-namespace \
  --set llm.provider=openai \
  --set llm.apiKey=sk-your-api-key \
  --set storage.postgres.enabled=true
```

### 3. Cloud Platform Deployment

#### AWS Deployment

**ECS with Fargate:**
```json
{
  "family": "lightrag-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "lightrag",
      "image": "your-account.dkr.ecr.region.amazonaws.com/lightrag:latest",
      "portMappings": [
        {
          "containerPort": 9621,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LLM_BINDING",
          "value": "openai"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:lightrag-secrets"
        }
      ]
    }
  ]
}
```

**EKS Deployment:**
```bash
# Create EKS cluster
eksctl create cluster --name lightrag-cluster --region us-west-2

# Deploy using Kubernetes manifests
kubectl apply -f k8s-deploy/
```

#### Google Cloud Platform

**Cloud Run Deployment:**
```bash
# Build and push container
gcloud builds submit --tag gcr.io/PROJECT-ID/lightrag

# Deploy to Cloud Run
gcloud run deploy lightrag \
  --image gcr.io/PROJECT-ID/lightrag \
  --platform managed \
  --region us-central1 \
  --set-env-vars LLM_BINDING=openai \
  --set-env-vars OPENAI_API_KEY=sk-your-key
```

**GKE Deployment:**
```bash
# Create GKE cluster
gcloud container clusters create lightrag-cluster \
  --zone us-central1-a \
  --num-nodes 3

# Deploy using Helm or kubectl
helm install lightrag ./charts/lightrag
```

#### Azure Deployment

**Container Instances:**
```bash
az container create \
  --resource-group lightrag-rg \
  --name lightrag-container \
  --image lightrag:latest \
  --ports 9621 \
  --environment-variables LLM_BINDING=openai \
  --secure-environment-variables OPENAI_API_KEY=sk-your-key
```

**AKS Deployment:**
```bash
# Create AKS cluster
az aks create \
  --resource-group lightrag-rg \
  --name lightrag-cluster \
  --node-count 3

# Deploy using kubectl
kubectl apply -f k8s-deploy/
```

## Configuration Management

### Environment Variables

**Core Configuration:**
```bash
# LLM Provider
LLM_BINDING=openai               # openai, ollama, azure_openai, xai
LLM_MODEL=gpt-4                  # Model name
OPENAI_API_KEY=sk-...           # API key

# Embedding Provider
EMBEDDING_BINDING=openai         # Provider for embeddings
EMBEDDING_MODEL=text-embedding-ada-002

# Storage Configuration
KV_STORAGE=postgres             # postgres, redis, mongo, json
VECTOR_STORAGE=pgvector         # pgvector, milvus, qdrant, nano
GRAPH_STORAGE=postgres          # postgres, neo4j, networkx
DOC_STATUS_STORAGE=postgres     # postgres, mongo, json

# Server Configuration
PORT=9621                       # API server port
HOST=0.0.0.0                   # Bind address
WORKERS=4                       # Gunicorn workers

# Performance
MAX_ASYNC=5                     # Concurrent operations
TIMEOUT=120                     # Request timeout (seconds)

# Security
AUTH_ENABLED=true               # Enable authentication
JWT_SECRET_KEY=your-secret      # JWT signing key (32+ chars)
JWT_EXPIRE_HOURS=24            # Token expiration
RATE_LIMIT_ENABLED=true        # Enable rate limiting

# Database (if using PostgreSQL)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lightrag
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=secure_password

# Redis (if using Redis)
REDIS_URL=redis://localhost:6379/0
```

### Configuration Templates

**Development Template:**
```bash
# Development optimized for quick iteration
LLM_BINDING=ollama
LLM_MODEL=llama2:7b
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-text

# Lightweight storage
KV_STORAGE=json
VECTOR_STORAGE=nano
GRAPH_STORAGE=networkx

# Development settings
AUTH_ENABLED=false
DEBUG=true
WORKERS=1
```

**Production Template:**
```bash
# Production optimized for performance and security
LLM_BINDING=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-production-key

# Production storage
KV_STORAGE=postgres
VECTOR_STORAGE=pgvector
GRAPH_STORAGE=postgres

# Security enabled
AUTH_ENABLED=true
JWT_SECRET_KEY=production-secret-key-min-32-characters
RATE_LIMIT_ENABLED=true

# Performance optimized
WORKERS=4
MAX_ASYNC=5
```

### Secrets Management

**Docker Compose Secrets:**
```yaml
services:
  lightrag:
    environment:
      - OPENAI_API_KEY_FILE=/run/secrets/openai_key
    secrets:
      - openai_key

secrets:
  openai_key:
    external: true
```

**Kubernetes Secrets:**
```bash
# Create secret
kubectl create secret generic lightrag-secrets \
  --from-literal=openai-api-key=sk-your-key \
  --namespace lightrag

# Reference in deployment
env:
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: lightrag-secrets
      key: openai-api-key
```

## Database Setup

### PostgreSQL (Recommended for Production)

**Standard PostgreSQL:**
```bash
# Using Docker
docker run --name postgres \
  -e POSTGRES_DB=lightrag \
  -e POSTGRES_USER=lightrag \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  -d postgres:16
```

**Enhanced PostgreSQL with pgvector:**
```bash
# Use pre-built enhanced image
docker compose -f docker-compose.enhanced.yml up postgres

# Features:
# - PostgreSQL 16
# - pgvector extension for vector operations
# - Apache AGE for graph operations
# - Performance optimizations
```

**Manual PostgreSQL Setup:**
```sql
-- Create database
CREATE DATABASE lightrag;

-- Create user
CREATE USER lightrag WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE lightrag TO lightrag;

-- Install extensions (if available)
\c lightrag;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
```

### Redis (Optional, for Caching)

```bash
# Redis for caching and session storage
docker run --name redis \
  -p 6379:6379 \
  -d redis:7-alpine redis-server --appendonly yes
```

### Neo4j (Optional, for Advanced Graph Operations)

```bash
# Neo4j for complex graph queries
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -d neo4j:5
```

## Load Balancing & High Availability

### Nginx Load Balancer

```nginx
# /etc/nginx/sites-available/lightrag
upstream lightrag_backend {
    least_conn;
    server lightrag-1:9621;
    server lightrag-2:9621;
    server lightrag-3:9621;
}

server {
    listen 80;
    server_name lightrag.yourdomain.com;

    location / {
        proxy_pass http://lightrag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### HAProxy Configuration

```
# /etc/haproxy/haproxy.cfg
backend lightrag_servers
    balance roundrobin
    option httpchk GET /health
    server lightrag1 lightrag-1:9621 check
    server lightrag2 lightrag-2:9621 check
    server lightrag3 lightrag-3:9621 check

frontend lightrag_frontend
    bind *:80
    default_backend lightrag_servers
```

### Kubernetes High Availability

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lightrag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lightrag
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Monitoring & Observability

### Health Checks

**Basic Health Check:**
```bash
#!/bin/bash
# health-check.sh
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9621/health)
if [ $response = "200" ]; then
    echo "LightRAG is healthy"
    exit 0
else
    echo "LightRAG is unhealthy (HTTP $response)"
    exit 1
fi
```

**Comprehensive Health Check:**
```bash
# Check all components
curl http://localhost:9621/api/health | jq '.storage, .llm'
```

### Logging Configuration

**Structured Logging:**
```python
# Configure in production.env
LOG_LEVEL=INFO
LOG_FORMAT=json
AUDIT_LOGGING=true
```

**Log Aggregation (ELK Stack):**
```yaml
# docker-compose.monitoring.yml
services:
  elasticsearch:
    image: elastic/elasticsearch:8.0.0
  
  logstash:
    image: elastic/logstash:8.0.0
    
  kibana:
    image: elastic/kibana:8.0.0
```

### Metrics Collection

**Prometheus Integration:**
```yaml
# Add to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

## Backup & Disaster Recovery

### Database Backup

**Automated PostgreSQL Backup:**
```bash
#!/bin/bash
# backup-postgres.sh
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker compose exec postgres pg_dump -U lightrag lightrag > \
  "$BACKUP_DIR/lightrag_backup_$DATE.sql"

# Retention (keep last 7 days)
find $BACKUP_DIR -name "lightrag_backup_*.sql" -mtime +7 -delete
```

**Restore from Backup:**
```bash
# Restore database
docker compose exec -i postgres psql -U lightrag -d lightrag < \
  /backups/lightrag_backup_20250115_100000.sql
```

### Volume Backup

**Backup RAG Storage:**
```bash
# Backup document storage
tar -czf rag_storage_backup_$(date +%Y%m%d).tar.gz rag_storage/

# Backup to S3 (if using AWS)
aws s3 cp rag_storage_backup_$(date +%Y%m%d).tar.gz \
  s3://your-backup-bucket/lightrag-backups/
```

## Security Hardening

### Container Security

**Dockerfile Security Best Practices:**
```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' --uid 1001 lightrag
USER 1001

# Read-only root filesystem
# Set in docker-compose.yml:
# read_only: true
# tmpfs:
#   - /tmp

# Drop capabilities
# security_opt:
#   - no-new-privileges:true
```

### Network Security

**Docker Network Isolation:**
```yaml
# docker-compose.yml
networks:
  lightrag_internal:
    driver: bridge
    internal: true
  lightrag_external:
    driver: bridge

services:
  lightrag:
    networks:
      - lightrag_external
      - lightrag_internal
  
  postgres:
    networks:
      - lightrag_internal  # Only internal access
```

### SSL/TLS Configuration

**Nginx SSL Termination:**
```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/lightrag.crt;
    ssl_certificate_key /etc/ssl/private/lightrag.key;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://lightrag_backend;
    }
}
```

## Troubleshooting

### Common Issues

**1. Container Won't Start**
```bash
# Check logs
docker compose logs lightrag

# Common causes:
# - Missing environment variables
# - Port conflicts
# - Invalid configuration
```

**2. Database Connection Issues**
```bash
# Test database connectivity
docker compose exec lightrag python -c "
import psycopg2
conn = psycopg2.connect(
    host='postgres',
    database='lightrag',
    user='lightrag',
    password='password'
)
print('Database connected successfully')
"
```

**3. LLM API Issues**
```bash
# Test LLM connectivity
curl -X POST "http://localhost:9621/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "mode": "naive"}'
```

**4. Performance Issues**
```bash
# Monitor resource usage
docker stats

# Check concurrent operations
# Reduce MAX_ASYNC if necessary
```

### Debug Mode

**Enable Debug Logging:**
```bash
# Add to .env
DEBUG=true
LOG_LEVEL=DEBUG

# Restart services
docker compose restart lightrag
```

**Health Check Debug:**
```bash
# Detailed health information
curl http://localhost:9621/api/health | jq '.'
```

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor system health
- Check error logs
- Verify backup completion

**Weekly:**
- Review performance metrics
- Update security patches
- Clean up old logs

**Monthly:**
- Performance optimization review
- Capacity planning
- Security audit

### Update Procedures

**Rolling Update (Kubernetes):**
```bash
# Update image
kubectl set image deployment/lightrag \
  lightrag=lightrag:v1.5.1 -n lightrag

# Monitor rollout
kubectl rollout status deployment/lightrag -n lightrag
```

**Blue-Green Deployment:**
```bash
# Deploy new version alongside current
docker compose -f docker-compose.blue-green.yml up green

# Switch traffic after verification
# Update load balancer configuration
```

This deployment guide provides comprehensive coverage for deploying LightRAG in various environments, from development to enterprise production deployments.