# xAI Integration Guide

## Overview

LightRAG provides full integration with xAI's Grok models, offering advanced conversational AI capabilities with specialized handling for optimal performance and reliability. This guide covers setup, configuration, and best practices for using xAI models in your LightRAG deployment.

## Prerequisites

### xAI Account Setup
1. **Create xAI Account**: Sign up at [x.ai](https://x.ai/)
2. **Generate API Key**: Navigate to API settings and create a new API key
3. **Verify Access**: Ensure you have access to the desired Grok models

### System Requirements
- **LightRAG Version**: 1.5.0 or higher
- **Python**: 3.10+ with async support
- **Network**: Stable internet connection for API calls
- **Memory**: Minimum 4GB RAM (8GB recommended for production)

## Supported Models

### Available Grok Models
- **grok-3-mini**: Latest optimized model for fast responses
- **grok-2-1212**: High-performance general purpose model
- **grok-2-vision-1212**: Multimodal model with vision capabilities

### Model Capabilities
- **Context Length**: Up to 128K tokens
- **Languages**: Multi-language support with focus on English
- **Specialties**: Real-time information, conversational AI, reasoning
- **Rate Limits**: Varies by subscription tier

## Configuration

### Environment Variables

Add the following to your `.env` file:

```bash
# xAI Configuration
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
XAI_API_KEY=your-xai-api-key-here
XAI_API_BASE=https://api.x.ai/v1

# xAI-Specific Performance Settings
MAX_ASYNC=2                    # CRITICAL: Prevents timeout issues
TIMEOUT=240                    # 4 minutes for complex operations
LLM_REQUEST_TIMEOUT=180       # 3 minutes per request

# Optional: Embedding Model (use separate provider)
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
```

### Production Configuration

For production deployments, use these optimized settings:

```bash
# Production xAI Settings
LLM_BINDING=xai
LLM_MODEL=grok-2-1212
XAI_API_KEY=${XAI_API_KEY}
XAI_API_BASE=https://api.x.ai/v1

# Performance Optimization
MAX_ASYNC=2
TIMEOUT=300
LLM_REQUEST_TIMEOUT=240
RETRY_ATTEMPTS=3
RETRY_DELAY=2

# Reliability Settings
CACHE_ENABLED=true
CACHE_TTL=3600
HEALTH_CHECK_INTERVAL=30
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
services:
  lightrag:
    image: lightrag:latest
    environment:
      - LLM_BINDING=xai
      - LLM_MODEL=grok-3-mini
      - XAI_API_KEY=${XAI_API_KEY}
      - MAX_ASYNC=2
      - TIMEOUT=240
    secrets:
      - xai_api_key

secrets:
  xai_api_key:
    external: true
```

## Usage Examples

### Basic Python Usage

```python
import asyncio
from lightrag import LightRAG

async def xai_example():
    # Initialize LightRAG with xAI
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_binding="xai",
        llm_model="grok-3-mini",
        api_key="your-xai-api-key"
    )
    
    # Initialize storage
    await rag.initialize_storages()
    
    try:
        # Insert document
        result = await rag.ainsert(
            "xAI's Grok models are advanced conversational AI systems..."
        )
        print(f"Document inserted: {result}")
        
        # Query with hybrid mode (recommended)
        response = await rag.aquery(
            "What are the key features of Grok models?",
            mode="hybrid"
        )
        print(f"Response: {response}")
        
    finally:
        await rag.finalize_storages()

# Run example
asyncio.run(xai_example())
```

### Advanced Configuration

```python
from lightrag import LightRAG
from lightrag.llm import xAILLM

async def advanced_xai_setup():
    # Custom xAI configuration
    llm = xAILLM(
        api_key="your-api-key",
        model="grok-2-1212",
        max_async=2,
        timeout=240,
        retry_attempts=3
    )
    
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model=llm,
        # Use separate embedding provider
        embedding_binding="ollama",
        embedding_model="bge-m3:latest"
    )
    
    await rag.initialize_storages()
    
    # Process multiple documents with error handling
    documents = [
        "Document 1 content...",
        "Document 2 content...",
        "Document 3 content..."
    ]
    
    for i, doc in enumerate(documents):
        try:
            result = await rag.ainsert(doc)
            print(f"Document {i+1} processed successfully")
        except Exception as e:
            print(f"Error processing document {i+1}: {e}")
            continue
    
    await rag.finalize_storages()
```

### REST API Usage

```bash
# Configure API server with xAI
export LLM_BINDING=xai
export LLM_MODEL=grok-3-mini
export XAI_API_KEY=your-api-key
export MAX_ASYNC=2

# Start API server
lightrag-server

# Upload document
curl -X POST "http://localhost:9621/documents/text" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "xAI Grok models provide advanced AI capabilities...",
    "metadata": {"source": "xai_info"}
  }'

# Query with xAI
curl -X POST "http://localhost:9621/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain xAI capabilities",
    "mode": "hybrid",
    "top_k": 5
  }'
```

## Best Practices

### Performance Optimization

**1. Async Concurrency Management**
```python
# CRITICAL: Always use MAX_ASYNC=2 for xAI
# Higher values may cause timeout issues
MAX_ASYNC = 2
```

**2. Request Timeout Configuration**
```python
# Conservative timeout settings
TIMEOUT = 240  # 4 minutes total
LLM_REQUEST_TIMEOUT = 180  # 3 minutes per request
```

**3. Retry Logic Implementation**
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def robust_xai_query(rag, query):
    """Robust querying with retry logic."""
    try:
        return await rag.aquery(query, mode="hybrid")
    except Exception as e:
        print(f"Query attempt failed: {e}")
        raise
```

### Error Handling

**Common Error Patterns:**
```python
import logging

async def handle_xai_errors(rag, content):
    """Comprehensive error handling for xAI operations."""
    try:
        result = await rag.ainsert(content)
        return result
        
    except TimeoutError:
        logging.warning("xAI request timed out, retrying with longer timeout")
        # Implement retry logic
        
    except ValueError as e:
        if "rate limit" in str(e).lower():
            logging.warning("Rate limit hit, waiting before retry")
            await asyncio.sleep(60)
            # Implement backoff strategy
        else:
            logging.error(f"Validation error: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Unexpected xAI error: {e}")
        # Implement fallback strategy
        raise
```

### Memory Management

```python
async def memory_efficient_processing(documents):
    """Process documents with memory efficiency."""
    rag = LightRAG(llm_binding="xai", llm_model="grok-3-mini")
    await rag.initialize_storages()
    
    try:
        # Process in batches to manage memory
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                await rag.ainsert(doc)
                
            # Optional: Clear cache between batches
            if hasattr(rag, 'clear_cache'):
                rag.clear_cache()
                
    finally:
        await rag.finalize_storages()
```

## Demo Scripts

LightRAG includes several demo scripts optimized for xAI:

### 1. Basic Demo
```bash
cd LightRAG
python examples/lightrag_xai_demo.py
```

### 2. Timeout-Resistant Demo (Recommended)
```bash
cd LightRAG
python examples/lightrag_xai_demo_timeout_fix.py
```

### 3. Robust Production Demo
```bash
cd LightRAG
python examples/lightrag_xai_demo_robust.py
```

### 4. Connection Test
```bash
cd LightRAG
python examples/test_xai_basic.py
```

## Troubleshooting

### Common Issues and Solutions

**1. Timeout Errors**
```
Error: Request timed out
```

**Solution:**
- Ensure `MAX_ASYNC=2` is set
- Increase `TIMEOUT` and `LLM_REQUEST_TIMEOUT`
- Use the timeout-resistant demo script

**2. Rate Limiting**
```
Error: Rate limit exceeded
```

**Solution:**
- Implement exponential backoff
- Reduce concurrent requests
- Consider upgrading xAI subscription

**3. API Key Issues**
```
Error: Unauthorized or invalid API key
```

**Solution:**
- Verify API key in xAI dashboard
- Check environment variable spelling
- Ensure API key has necessary permissions

**4. Model Not Found**
```
Error: Model not available
```

**Solution:**
- Verify model name spelling
- Check subscription access to model
- Use alternative model (grok-3-mini)

### Diagnostic Commands

**Test xAI Connection:**
```python
import asyncio
from lightrag.llm import xAILLM

async def test_connection():
    llm = xAILLM(api_key="your-key", model="grok-3-mini")
    try:
        response = await llm.agenerate([
            {"role": "user", "content": "Hello, this is a test."}
        ])
        print(f"Connection successful: {response}")
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_connection())
```

**Check Configuration:**
```bash
# Verify environment variables
echo "LLM_BINDING: $LLM_BINDING"
echo "LLM_MODEL: $LLM_MODEL"
echo "XAI_API_KEY: ${XAI_API_KEY:0:10}..." # Show first 10 chars
echo "MAX_ASYNC: $MAX_ASYNC"
```

### Performance Monitoring

**Monitor Response Times:**
```python
import time

async def monitor_performance(rag, queries):
    """Monitor xAI query performance."""
    results = []
    
    for query in queries:
        start_time = time.time()
        try:
            response = await rag.aquery(query, mode="hybrid")
            end_time = time.time()
            
            results.append({
                "query": query,
                "response_time": end_time - start_time,
                "success": True,
                "response_length": len(response)
            })
        except Exception as e:
            end_time = time.time()
            results.append({
                "query": query,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e)
            })
    
    return results
```

## Production Deployment

### Docker Configuration

**Dockerfile with xAI:**
```dockerfile
FROM python:3.12-slim

# Install LightRAG
COPY . /app
WORKDIR /app
RUN pip install -e ".[api]"

# xAI-specific environment
ENV LLM_BINDING=xai
ENV MAX_ASYNC=2
ENV TIMEOUT=240

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:9621/health || exit 1

EXPOSE 9621
CMD ["lightrag-server"]
```

### Kubernetes Deployment

```yaml
# xai-lightrag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag-xai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lightrag-xai
  template:
    metadata:
      labels:
        app: lightrag-xai
    spec:
      containers:
      - name: lightrag
        image: lightrag:latest
        env:
        - name: LLM_BINDING
          value: "xai"
        - name: LLM_MODEL
          value: "grok-3-mini"
        - name: MAX_ASYNC
          value: "2"
        - name: TIMEOUT
          value: "240"
        - name: XAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: xai-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9621
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 9621
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Secret
metadata:
  name: xai-secrets
type: Opaque
stringData:
  api-key: your-xai-api-key-here
```

### Monitoring and Alerting

```python
# monitoring/xai_monitor.py
import asyncio
import logging
from datetime import datetime

class xAIMonitor:
    """Monitor xAI integration health and performance."""
    
    def __init__(self, rag_instance):
        self.rag = rag_instance
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "last_success": None,
            "last_failure": None
        }
    
    async def health_check(self):
        """Perform periodic health check."""
        try:
            start_time = datetime.now()
            
            response = await self.rag.aquery(
                "Test query for health check",
                mode="naive"
            )
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            self.metrics["total_requests"] += 1
            self.metrics["successful_requests"] += 1
            self.metrics["last_success"] = end_time
            
            # Update average response time
            self._update_avg_response_time(response_time)
            
            logging.info(f"xAI health check passed: {response_time:.2f}s")
            return True
            
        except Exception as e:
            self.metrics["total_requests"] += 1
            self.metrics["failed_requests"] += 1
            self.metrics["last_failure"] = datetime.now()
            
            logging.error(f"xAI health check failed: {e}")
            return False
    
    def _update_avg_response_time(self, new_time):
        """Update average response time."""
        if self.metrics["avg_response_time"] == 0:
            self.metrics["avg_response_time"] = new_time
        else:
            # Simple moving average
            self.metrics["avg_response_time"] = (
                self.metrics["avg_response_time"] * 0.9 + new_time * 0.1
            )
    
    def get_health_status(self):
        """Get current health status."""
        success_rate = (
            self.metrics["successful_requests"] / 
            max(self.metrics["total_requests"], 1)
        )
        
        return {
            "success_rate": success_rate,
            "avg_response_time": self.metrics["avg_response_time"],
            "total_requests": self.metrics["total_requests"],
            "last_success": self.metrics["last_success"],
            "last_failure": self.metrics["last_failure"],
            "status": "healthy" if success_rate > 0.95 else "degraded"
        }
```

## Migration Guide

### From OpenAI to xAI

**1. Update Configuration:**
```bash
# Old OpenAI configuration
LLM_BINDING=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# New xAI configuration
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
XAI_API_KEY=your-xai-key
MAX_ASYNC=2  # Add this for xAI
```

**2. Code Changes:**
```python
# Minimal code changes required
# Most LightRAG code remains the same

# Old
rag = LightRAG(llm_binding="openai", llm_model="gpt-4")

# New
rag = LightRAG(llm_binding="xai", llm_model="grok-3-mini")
```

**3. Performance Tuning:**
- Reduce MAX_ASYNC from 5 to 2
- Increase timeouts for stability
- Monitor response times and adjust

### From Ollama to xAI

**Configuration Changes:**
```bash
# Old Ollama configuration
LLM_BINDING=ollama
LLM_MODEL=llama2:7b
OLLAMA_URL=http://localhost:11434

# New xAI configuration
LLM_BINDING=xai
LLM_MODEL=grok-3-mini
XAI_API_KEY=your-xai-key
MAX_ASYNC=2
```

## Security Considerations

### API Key Management

**1. Environment Variables:**
```bash
# Use environment variables, never hardcode
export XAI_API_KEY=your-api-key
```

**2. Secret Management (Kubernetes):**
```bash
kubectl create secret generic xai-secrets \
  --from-literal=api-key=your-xai-api-key
```

**3. Docker Secrets:**
```yaml
services:
  lightrag:
    secrets:
      - xai_api_key
    environment:
      - XAI_API_KEY_FILE=/run/secrets/xai_api_key

secrets:
  xai_api_key:
    external: true
```

### Network Security

**1. API Endpoint Restriction:**
```bash
# Only allow connections to xAI endpoints
XAI_API_BASE=https://api.x.ai/v1  # Official endpoint only
```

**2. Request Logging:**
```python
import logging

# Log requests without exposing API keys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xai_requests")

# Mask sensitive data in logs
def safe_log_request(request_data):
    masked_data = request_data.copy()
    if "api_key" in masked_data:
        masked_data["api_key"] = "***MASKED***"
    logger.info(f"xAI request: {masked_data}")
```

## Support and Resources

### Official Resources
- **xAI Documentation**: [docs.x.ai](https://docs.x.ai/)
- **API Reference**: [api.x.ai/docs](https://api.x.ai/docs)
- **Status Page**: [status.x.ai](https://status.x.ai/)

### Community and Support
- **LightRAG Issues**: Report xAI-specific issues on GitHub
- **xAI Community**: Join xAI developer communities
- **Discord/Forums**: Participate in AI/ML developer forums

### Additional Examples
- **GitHub Repository**: Complete example implementations
- **Demo Videos**: Video tutorials for xAI integration
- **Blog Posts**: Advanced use cases and optimizations

---

**Ready to get started with xAI?** Follow the configuration steps above and run the timeout-resistant demo script to validate your setup.

For production deployments, review the performance optimization and monitoring sections to ensure reliable operation.