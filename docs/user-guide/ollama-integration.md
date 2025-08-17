# Ollama Integration Guide

## Overview

Ollama provides local LLM and embedding model deployment, offering privacy, cost-effectiveness, and performance benefits for LightRAG. This guide covers complete setup, configuration, and security hardening for Ollama integration.

## Why Ollama for LightRAG?

### Privacy and Security Benefits
- **Local Processing**: All data stays on your infrastructure
- **No External API Calls**: Complete privacy for sensitive documents
- **GDPR/HIPAA Compliance**: Easier compliance with data protection regulations
- **Network Independence**: Works without internet connectivity

### Cost and Performance Benefits
- **No API Costs**: Eliminate per-token charges
- **Predictable Scaling**: Fixed infrastructure costs
- **Low Latency**: Local processing reduces response times
- **Bulk Processing**: No rate limiting constraints

### LightRAG Integration Features
- **Embedding Models**: High-quality local embeddings
- **LLM Models**: Optional local LLM processing
- **Async Support**: Full async/await integration
- **Batch Processing**: Efficient bulk operations

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: 8GB minimum (16GB+ recommended for production)
- **Storage**: 10GB+ for models (varies by model size)
- **GPU**: Optional but recommended (NVIDIA with CUDA support)
- **CPU**: Modern multi-core processor

### Network Requirements
- **Internet**: Required for initial model downloads
- **Bandwidth**: 1-10GB per model download
- **Internal Network**: LightRAG to Ollama connectivity

## Installation

### 1. Install Ollama

**Linux/macOS:**
```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

**Windows (WSL2):**
```bash
# Install in WSL2 environment
curl -fsSL https://ollama.ai/install.sh | sh

# Or download Windows installer from ollama.ai
```

**Docker Installation:**
```bash
# Pull Ollama Docker image
docker pull ollama/ollama

# Run Ollama container
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama

# With GPU support (NVIDIA)
docker run -d \
  --gpus all \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

### 2. Download Embedding Models

**Recommended Embedding Models:**

```bash
# BGE-M3 (Multilingual, 1024 dimensions) - RECOMMENDED
ollama pull bge-m3

# Nomic Embed Text (English, 768 dimensions)
ollama pull nomic-embed-text

# All-MiniLM (English, 384 dimensions) - Lightweight
ollama pull all-minilm

# BGE Large (English/Chinese, 1024 dimensions) - High quality
ollama pull bge-large

# E5 Large (Multilingual, 1024 dimensions)
ollama pull e5-large
```

**Model Comparison:**

| Model | Size | Dimensions | Languages | Best For |
|-------|------|------------|-----------|----------|
| **bge-m3** | 2.4GB | 1024 | 100+ | Production (recommended) |
| nomic-embed-text | 274MB | 768 | English | Development |
| all-minilm | 45MB | 384 | English | Testing/Low resource |
| bge-large | 1.3GB | 1024 | EN/ZH | High quality English |
| e5-large | 1.3GB | 1024 | 100+ | Multilingual production |

### 3. Verify Installation

```bash
# Check Ollama service
ollama list

# Test embedding generation
ollama embeddings nomic-embed-text "This is a test sentence"

# Check API endpoint
curl http://localhost:11434/api/version
```

## Configuration

### 1. LightRAG Configuration

**Basic Setup (.env):**
```bash
# Ollama Embedding Configuration
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024

# Ollama Connection
OLLAMA_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_MAX_RETRIES=3

# Optional: Use Ollama for LLM as well
LLM_BINDING=ollama
LLM_MODEL=llama2:7b

# Storage Configuration
KV_STORAGE=postgres
VECTOR_STORAGE=pgvector
GRAPH_STORAGE=postgres
```

**Production Configuration (.env):**
```bash
# Production Ollama Settings
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024

# Connection Settings
OLLAMA_URL=http://ollama-server:11434
OLLAMA_TIMEOUT=600
OLLAMA_MAX_RETRIES=5
OLLAMA_BATCH_SIZE=32

# Performance Settings
OLLAMA_NUM_PARALLEL=4
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=100

# Security Settings
OLLAMA_ENABLE_CORS=false
OLLAMA_ALLOWED_ORIGINS=localhost,127.0.0.1
```

### 2. Docker Compose Integration

**Development Stack:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 3

  lightrag:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - EMBEDDING_BINDING=ollama
      - EMBEDDING_MODEL=bge-m3:latest
      - OLLAMA_URL=http://ollama:11434
    volumes:
      - ./rag_storage:/app/rag_storage

volumes:
  ollama_data:
```

**Production Stack with GPU:**
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "127.0.0.1:11434:11434"  # Bind to localhost only
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS=lightrag
      - OLLAMA_NUM_PARALLEL=4
    networks:
      - lightrag_internal
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  lightrag:
    build: .
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - EMBEDDING_BINDING=ollama
      - EMBEDDING_MODEL=bge-m3:latest
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_TIMEOUT=600
    networks:
      - lightrag_internal
      - lightrag_external
    restart: unless-stopped

networks:
  lightrag_internal:
    driver: bridge
    internal: true
  lightrag_external:
    driver: bridge

volumes:
  ollama_data:
    driver: local
```

### 3. Kubernetes Deployment

**Ollama Deployment:**
```yaml
# ollama-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama
  namespace: lightrag
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
        env:
        - name: OLLAMA_HOST
          value: "0.0.0.0"
        - name: OLLAMA_NUM_PARALLEL
          value: "4"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: ollama-data
          mountPath: /root/.ollama
        livenessProbe:
          httpGet:
            path: /api/version
            port: 11434
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/version
            port: 11434
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: ollama-data
        persistentVolumeClaim:
          claimName: ollama-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-service
  namespace: lightrag
spec:
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ollama-pvc
  namespace: lightrag
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

## Usage Examples

### 1. Python SDK Usage

**Basic Embedding Generation:**
```python
import asyncio
from lightrag import LightRAG

async def ollama_embedding_example():
    # Initialize LightRAG with Ollama embeddings
    rag = LightRAG(
        working_dir="./rag_storage",
        embedding_binding="ollama",
        embedding_model="bge-m3:latest",
        ollama_url="http://localhost:11434"
    )
    
    await rag.initialize_storages()
    
    try:
        # Insert document (will use Ollama for embeddings)
        result = await rag.ainsert(
            "Ollama provides local LLM deployment capabilities..."
        )
        print(f"Document inserted: {result}")
        
        # Query using Ollama embeddings
        response = await rag.aquery(
            "What are the benefits of local LLM deployment?",
            mode="hybrid"
        )
        print(f"Response: {response}")
        
    finally:
        await rag.finalize_storages()

# Run example
asyncio.run(ollama_embedding_example())
```

**Advanced Configuration:**
```python
from lightrag import LightRAG
from lightrag.llm import OllamaLLM

async def advanced_ollama_setup():
    # Custom Ollama configuration
    embedding_config = {
        "url": "http://ollama-cluster:11434",
        "model": "bge-m3:latest",
        "timeout": 600,
        "batch_size": 32,
        "max_retries": 5
    }
    
    rag = LightRAG(
        working_dir="./rag_storage",
        embedding_binding="ollama",
        embedding_config=embedding_config,
        # Optional: Use different LLM
        llm_binding="openai",
        llm_model="gpt-4"
    )
    
    await rag.initialize_storages()
    
    # Batch processing with Ollama embeddings
    documents = [
        "Document 1 content...",
        "Document 2 content...",
        "Document 3 content..."
    ]
    
    for doc in documents:
        try:
            await rag.ainsert(doc)
            print("Document processed successfully")
        except Exception as e:
            print(f"Error processing document: {e}")
    
    await rag.finalize_storages()
```

### 2. REST API Usage

**Configure API Server:**
```bash
# Start API server with Ollama embeddings
export EMBEDDING_BINDING=ollama
export EMBEDDING_MODEL=bge-m3:latest
export OLLAMA_URL=http://localhost:11434

lightrag-server
```

**API Requests:**
```bash
# Upload document (will use Ollama for embeddings)
curl -X POST "http://localhost:9621/documents/text" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Ollama enables local LLM deployment...",
    "metadata": {"source": "ollama_guide"}
  }'

# Query with Ollama embeddings
curl -X POST "http://localhost:9621/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does local LLM deployment work?",
    "mode": "hybrid",
    "top_k": 5
  }'
```

## Performance Optimization

### 1. Model Selection Strategy

**Development Environment:**
```bash
# Lightweight model for development
EMBEDDING_MODEL=all-minilm:latest
EMBEDDING_DIM=384
```

**Production Environment:**
```bash
# High-quality multilingual model
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
```

**High-Performance Setup:**
```bash
# Optimized for speed and quality
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
OLLAMA_NUM_PARALLEL=8
OLLAMA_BATCH_SIZE=64
```

### 2. Hardware Optimization

**CPU Optimization:**
```bash
# Optimize for CPU-only deployment
export OLLAMA_NUM_THREAD=8  # Match CPU cores
export OLLAMA_USE_MLOCK=true  # Lock memory for performance
```

**GPU Optimization:**
```bash
# NVIDIA GPU optimization
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_LAYERS=32  # Offload layers to GPU

# Multiple GPU setup
export CUDA_VISIBLE_DEVICES=0,1
export OLLAMA_PARALLEL_REQUESTS=4
```

**Memory Optimization:**
```bash
# Optimize memory usage
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=300  # Keep models loaded for 5 minutes
export OLLAMA_HOST=127.0.0.1  # Reduce network overhead
```

### 3. Batch Processing

**Optimize Batch Size:**
```python
# Benchmark different batch sizes
import time
import asyncio
from lightrag.llm import OllamaEmbedding

async def benchmark_batch_sizes():
    """Benchmark different batch sizes for optimal performance."""
    
    ollama = OllamaEmbedding(url="http://localhost:11434")
    texts = ["Sample text " + str(i) for i in range(100)]
    
    batch_sizes = [1, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            await ollama.aembedding(batch)
        
        end_time = time.time()
        throughput = len(texts) / (end_time - start_time)
        
        print(f"Batch size {batch_size}: {throughput:.2f} texts/second")

asyncio.run(benchmark_batch_sizes())
```

## Security Hardening

### 1. Network Security

**Firewall Configuration:**
```bash
# Allow only specific IP ranges
sudo ufw allow from 10.0.0.0/8 to any port 11434
sudo ufw allow from 172.16.0.0/12 to any port 11434
sudo ufw allow from 192.168.0.0/16 to any port 11434

# Deny all other access
sudo ufw deny 11434
```

**Nginx Reverse Proxy:**
```nginx
# /etc/nginx/sites-available/ollama
upstream ollama_backend {
    server 127.0.0.1:11434;
}

server {
    listen 443 ssl http2;
    server_name ollama.internal.company.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/ollama.crt;
    ssl_certificate_key /etc/ssl/private/ollama.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=ollama:10m rate=10r/s;
    limit_req zone=ollama burst=20 nodelay;
    
    # IP whitelist
    allow 10.0.0.0/8;
    allow 172.16.0.0/12;
    allow 192.168.0.0/16;
    deny all;
    
    location / {
        proxy_pass http://ollama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        
        # Request size limits
        client_max_body_size 10M;
    }
}
```

### 2. Container Security

**Security-Hardened Dockerfile:**
```dockerfile
FROM ollama/ollama:latest

# Create non-root user with UID >2000
RUN groupadd -g 2002 ollama_secure && \
    useradd -u 2002 -g ollama_secure -s /bin/bash -m ollama_secure

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Create secure directories
RUN mkdir -p /opt/ollama/models && \
    chown -R ollama_secure:ollama_secure /opt/ollama && \
    chmod 750 /opt/ollama

# Security configurations
COPY ollama-secure.conf /etc/ollama/config.json
RUN chmod 600 /etc/ollama/config.json

USER ollama_secure
WORKDIR /opt/ollama

EXPOSE 11434

CMD ["ollama", "serve"]
```

**Docker Compose Security:**
```yaml
# docker-compose.production.yml
services:
  ollama:
    build:
      context: ./ollama
      dockerfile: Dockerfile.secure
    user: "2002:2002"
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS=lightrag
    volumes:
      - ollama_models:/opt/ollama/models:Z
    networks:
      - ollama_internal
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp
      - /var/log
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/version"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  ollama_internal:
    driver: bridge
    internal: true

volumes:
  ollama_models:
    driver: local
```

### 3. Access Control

**API Key Authentication (Custom Implementation):**
```python
# ollama_auth_proxy.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import os

app = FastAPI()
security = HTTPBearer()

VALID_API_KEYS = {
    os.getenv("OLLAMA_API_KEY_1", ""),
    os.getenv("OLLAMA_API_KEY_2", ""),
}

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.post("/api/embeddings")
async def proxy_embeddings(
    request: dict,
    api_key: str = Depends(verify_api_key)
):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/embeddings",
            json=request,
            timeout=600
        )
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11435)
```

### 4. Model Security

**Model Verification:**
```bash
#!/bin/bash
# verify-ollama-models.sh

MODELS_DIR="/root/.ollama/models"
CHECKSUM_FILE="/opt/security/model-checksums.txt"

# Verify model integrity
while IFS= read -r line; do
    model_path=$(echo "$line" | cut -d' ' -f2)
    expected_checksum=$(echo "$line" | cut -d' ' -f1)
    
    if [ -f "$MODELS_DIR/$model_path" ]; then
        actual_checksum=$(sha256sum "$MODELS_DIR/$model_path" | cut -d' ' -f1)
        
        if [ "$actual_checksum" != "$expected_checksum" ]; then
            echo "WARNING: Model integrity check failed for $model_path"
            echo "Expected: $expected_checksum"
            echo "Actual: $actual_checksum"
            exit 1
        fi
    else
        echo "ERROR: Model not found: $model_path"
        exit 1
    fi
done < "$CHECKSUM_FILE"

echo "All models verified successfully"
```

## Monitoring and Observability

### 1. Health Monitoring

**Health Check Script:**
```bash
#!/bin/bash
# ollama-health-check.sh

OLLAMA_URL="http://localhost:11434"
MODEL_NAME="bge-m3:latest"

# Test API availability
if ! curl -f "$OLLAMA_URL/api/version" > /dev/null 2>&1; then
    echo "CRITICAL: Ollama API not responding"
    exit 2
fi

# Test model availability
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "CRITICAL: Model $MODEL_NAME not loaded"
    exit 2
fi

# Test embedding generation
RESPONSE=$(curl -s -X POST "$OLLAMA_URL/api/embeddings" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_NAME'",
        "prompt": "health check test"
    }')

if [ -z "$RESPONSE" ] || echo "$RESPONSE" | grep -q "error"; then
    echo "CRITICAL: Embedding generation failed"
    echo "Response: $RESPONSE"
    exit 2
fi

echo "OK: Ollama is healthy"
exit 0
```

### 2. Performance Monitoring

**Metrics Collection:**
```python
# ollama_metrics.py
import asyncio
import time
import logging
from datetime import datetime
import psutil
import httpx

class OllamaMonitor:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_response_time": 0,
            "model_memory_usage": 0,
            "last_health_check": None
        }
    
    async def check_health(self):
        """Perform health check and collect metrics."""
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ollama_url}/api/version",
                    timeout=10
                )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                self.metrics["requests_success"] += 1
                self.metrics["last_health_check"] = datetime.now()
                self._update_avg_response_time(response_time)
                
                # Check system resources
                process = self._find_ollama_process()
                if process:
                    self.metrics["model_memory_usage"] = process.memory_info().rss / 1024 / 1024  # MB
                
                logging.info(f"Ollama health check passed: {response_time:.3f}s")
                return True
            else:
                self.metrics["requests_failed"] += 1
                logging.error(f"Ollama health check failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.metrics["requests_failed"] += 1
            logging.error(f"Ollama health check error: {e}")
            return False
        finally:
            self.metrics["requests_total"] += 1
    
    def _find_ollama_process(self):
        """Find Ollama process for resource monitoring."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def _update_avg_response_time(self, new_time):
        """Update average response time."""
        if self.metrics["avg_response_time"] == 0:
            self.metrics["avg_response_time"] = new_time
        else:
            # Simple moving average
            self.metrics["avg_response_time"] = (
                self.metrics["avg_response_time"] * 0.9 + new_time * 0.1
            )
    
    def get_metrics(self):
        """Get current metrics."""
        success_rate = (
            self.metrics["requests_success"] / 
            max(self.metrics["requests_total"], 1)
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "status": "healthy" if success_rate > 0.95 else "degraded"
        }

# Usage
async def main():
    monitor = OllamaMonitor()
    
    while True:
        await monitor.check_health()
        metrics = monitor.get_metrics()
        print(f"Ollama Status: {metrics}")
        
        await asyncio.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Logging Configuration

**Structured Logging:**
```python
# ollama_logging.py
import logging
import json
from datetime import datetime

class OllamaJSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": "ollama",
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'model_name'):
            log_entry["model_name"] = record.model_name
        if hasattr(record, 'response_time'):
            log_entry["response_time"] = record.response_time
            
        return json.dumps(log_entry)

# Configure logging
def setup_ollama_logging():
    logger = logging.getLogger("ollama")
    logger.setLevel(logging.INFO)
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(OllamaJSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler for audit logs
    file_handler = logging.FileHandler("/var/log/ollama/audit.log")
    file_handler.setFormatter(OllamaJSONFormatter())
    logger.addHandler(file_handler)
    
    return logger
```

## Troubleshooting

### Common Issues

**1. Model Download Issues**
```bash
# Issue: Model download fails
# Solution: Check disk space and network
df -h /root/.ollama
curl -I https://ollama.ai

# Issue: Model not found
# Solution: List available models and download
ollama list
ollama pull bge-m3:latest
```

**2. Connection Issues**
```bash
# Issue: Cannot connect to Ollama
# Solution: Check service and port
systemctl status ollama
netstat -tulpn | grep 11434

# Issue: Timeout errors
# Solution: Increase timeout settings
export OLLAMA_TIMEOUT=600
```

**3. Performance Issues**
```bash
# Issue: Slow embedding generation
# Solution: Check resources and optimize
htop  # Check CPU/Memory usage
nvidia-smi  # Check GPU usage (if applicable)

# Optimize batch size
export OLLAMA_BATCH_SIZE=16  # Reduce if memory limited
```

**4. Memory Issues**
```bash
# Issue: Out of memory errors
# Solution: Reduce model concurrency
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=2

# Monitor memory usage
watch -n 1 'free -h && echo "Ollama Process:" && ps aux | grep ollama'
```

### Diagnostic Commands

**System Diagnostics:**
```bash
# Check Ollama installation
ollama --version
which ollama

# Check models
ollama list
du -sh /root/.ollama/models/*

# Test API endpoints
curl http://localhost:11434/api/version
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "prompt": "test"}'

# Check system resources
free -h
df -h
nvidia-smi  # If GPU available
```

**Performance Testing:**
```python
# performance_test.py
import asyncio
import time
import httpx

async def test_embedding_performance():
    """Test embedding generation performance."""
    
    test_texts = [
        "This is a test sentence for performance testing.",
        "Another test sentence with different content.",
        "Performance testing is important for production.",
        "Ollama provides local LLM capabilities.",
        "LightRAG integrates well with Ollama."
    ]
    
    async with httpx.AsyncClient() as client:
        # Warm up
        await client.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": test_texts[0]},
            timeout=60
        )
        
        # Performance test
        start_time = time.time()
        
        for text in test_texts:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "bge-m3", "prompt": text},
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(test_texts) / total_time
        
        print(f"Processed {len(test_texts)} texts in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} texts/second")
        print(f"Average time per text: {total_time/len(test_texts):.3f}s")

if __name__ == "__main__":
    asyncio.run(test_embedding_performance())
```

## Migration and Integration

### From OpenAI Embeddings to Ollama

**Configuration Migration:**
```bash
# Old OpenAI configuration
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_API_KEY=sk-...

# New Ollama configuration
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
OLLAMA_URL=http://localhost:11434
```

**Code Migration:**
```python
# Minimal code changes required
# Old setup
rag = LightRAG(
    embedding_binding="openai",
    embedding_model="text-embedding-ada-002"
)

# New setup
rag = LightRAG(
    embedding_binding="ollama",
    embedding_model="bge-m3:latest",
    ollama_url="http://localhost:11434"
)
```

### Hybrid Setup (Ollama + Cloud LLM)

**Recommended Configuration:**
```bash
# Use Ollama for embeddings (privacy + cost)
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
OLLAMA_URL=http://localhost:11434

# Use cloud LLM for generation (quality)
LLM_BINDING=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Or use xAI for generation
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
XAI_API_KEY=your-xai-key
```

## Production Deployment Checklist

### Pre-Production Validation

- [ ] **Model Installation**
  - [ ] Required models downloaded and verified
  - [ ] Model checksums validated
  - [ ] Sufficient disk space allocated

- [ ] **Performance Testing**
  - [ ] Embedding generation performance benchmarked
  - [ ] Batch processing optimized
  - [ ] Memory usage within limits
  - [ ] GPU utilization optimized (if applicable)

- [ ] **Security Hardening**
  - [ ] Network access restricted
  - [ ] Container security configured
  - [ ] API authentication implemented
  - [ ] Model integrity verified

- [ ] **Monitoring Setup**
  - [ ] Health checks configured
  - [ ] Performance monitoring enabled
  - [ ] Logging and alerting implemented
  - [ ] Resource monitoring active

- [ ] **High Availability**
  - [ ] Multiple Ollama instances deployed
  - [ ] Load balancing configured
  - [ ] Failover procedures tested
  - [ ] Backup and recovery tested

### Production Best Practices

1. **Resource Allocation**
   - Allocate 4-8GB RAM per Ollama instance
   - Use SSD storage for model files
   - Consider GPU acceleration for large deployments

2. **Model Management**
   - Version control model downloads
   - Automate model updates
   - Validate model integrity regularly

3. **Network Security**
   - Use internal networks for Ollama communication
   - Implement TLS termination at load balancer
   - Restrict API access to authorized clients

4. **Operational Excellence**
   - Implement comprehensive monitoring
   - Set up automated alerting
   - Document troubleshooting procedures
   - Test disaster recovery procedures

---

**Ready for Local Embeddings?** Follow this guide to deploy Ollama with LightRAG for privacy-preserving, cost-effective embedding generation.

For advanced deployments, consider the hybrid approach using Ollama for embeddings and cloud LLMs for text generation to optimize both privacy and quality.