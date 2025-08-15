# CI/CD Troubleshooting Guide for LightRAG

This guide provides solutions for common CI/CD issues and debugging procedures.

## Table of Contents

- [CI Pipeline Issues](#ci-pipeline-issues)
- [Deployment Problems](#deployment-problems)
- [Container Issues](#container-issues)
- [Database Problems](#database-problems)
- [Network and Connectivity](#network-and-connectivity)
- [Performance Issues](#performance-issues)
- [Security and Authentication](#security-and-authentication)
- [Monitoring and Logging](#monitoring-and-logging)
- [Debug Tools and Commands](#debug-tools-and-commands)

## CI Pipeline Issues

### Test Failures

#### Symptoms
- Tests failing in CI but passing locally
- Intermittent test failures
- Timeout errors in test execution

#### Common Causes and Solutions

**Environment Differences**:
```bash
# Check Python version consistency
python --version  # Local
# vs CI environment (check workflow file)

# Verify dependency versions
pip freeze > local-requirements.txt
# Compare with CI dependency installation
```

**Service Dependencies**:
```bash
# Check service health in CI
# Add debug steps to CI workflow:
- name: Debug services
  run: |
    docker ps
    curl -f http://localhost:5432 || echo "PostgreSQL not ready"
    redis-cli ping || echo "Redis not ready"
```

**Race Conditions**:
```bash
# Add explicit waits in CI
- name: Wait for services
  run: |
    until pg_isready -h localhost -p 5432; do
      echo "Waiting for PostgreSQL..."
      sleep 2
    done
```

**Resource Constraints**:
```bash
# Reduce parallelism in CI
pytest tests/ -n 1  # Instead of -n auto

# Split test execution
pytest tests/unit/ tests/integration/ -k "not slow"
```

#### Debug Steps

1. **Reproduce Locally**:
   ```bash
   # Use same Python version as CI
   pyenv install 3.10.12
   pyenv local 3.10.12
   
   # Install exact dependencies
   pip install -e ".[test]"
   
   # Run failing test
   pytest tests/path/to/failing_test.py -v -s
   ```

2. **Check CI Logs**:
   - Review full test output
   - Look for setup/teardown issues
   - Check service initialization

3. **Add Debug Output**:
   ```bash
   # Add to CI workflow
   - name: Debug environment
     run: |
       env | sort
       ps aux
       netstat -tlnp
   ```

### Build Failures

#### Docker Build Issues

**Symptoms**:
- Image build failures
- Layer caching problems
- Dependency installation errors

**Solutions**:
```bash
# Clear Docker cache
docker builder prune -a

# Build with no cache
docker build --no-cache -t lightrag:debug .

# Check base image availability
docker pull python:3.10-slim

# Debug multi-stage builds
docker build --target development -t lightrag:dev .
```

**Dependency Issues**:
```bash
# Pin dependency versions
pip install "requests>=2.28.0,<3.0.0"

# Use requirements.txt for reproducible builds
pip freeze > requirements-lock.txt
# Add to Dockerfile: COPY requirements-lock.txt requirements.txt
```

#### Package Build Issues

**Python Package Problems**:
```bash
# Check package metadata
python setup.py check

# Validate pyproject.toml
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"

# Test wheel creation
python -m build --wheel
twine check dist/*
```

### Linting and Formatting Failures

#### Code Quality Issues

**Ruff Failures**:
```bash
# Check specific rules
ruff check lightrag/ --select=E9,F63,F7,F82

# Auto-fix issues
ruff check lightrag/ --fix

# Disable specific rules if needed
# pyproject.toml: ignore = ["E501", "W503"]
```

**Import Sorting Issues**:
```bash
# Fix import order
isort lightrag/ --diff
isort lightrag/ --force-single-line

# Check configuration
isort --show-diff lightrag/
```

**Type Checking Problems**:
```bash
# Install missing type stubs
mypy lightrag/ --install-types

# Ignore specific modules
mypy lightrag/ --ignore-missing-imports
```

### Security Scan Failures

#### Vulnerability Issues

**Dependency Vulnerabilities**:
```bash
# Check specific vulnerabilities
safety check --json | jq '.vulnerabilities'

# Update vulnerable packages
pip install --upgrade package-name

# Ignore false positives
safety check --ignore 12345
```

**Secret Detection**:
```bash
# Check for secrets
grep -r "api_key\|password\|token" --include="*.py" .

# Remove secrets from git history
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/secret/file' \
  --prune-empty --tag-name-filter cat -- --all
```

## Deployment Problems

### Deployment Script Failures

#### Common Issues

**Permission Errors**:
```bash
# Make scripts executable
chmod +x scripts/deploy/*.sh

# Check Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

**Environment Variable Issues**:
```bash
# Verify required variables
echo "ENVIRONMENT: $ENVIRONMENT"
echo "IMAGE_TAG: $IMAGE_TAG"

# Check .env file
cat .env | grep -v "^#" | grep -v "^$"

# Source environment
set -a; source .env; set +a
```

**Image Pull Failures**:
```bash
# Check image exists
docker pull ghcr.io/hkuds/lightrag:v1.2.3

# Login to registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Check image manifest
docker manifest inspect ghcr.io/hkuds/lightrag:v1.2.3
```

#### Debug Steps

1. **Test Deployment Script Locally**:
   ```bash
   # Set debug mode
   set -x
   
   # Run script step by step
   source scripts/deploy/deploy-docker.sh
   # Comment out main() call, run functions individually
   ```

2. **Check Service Dependencies**:
   ```bash
   # Verify database connectivity
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -c "SELECT 1;"
   
   # Test Redis connection
   redis-cli -h $REDIS_HOST ping
   ```

3. **Validate Configuration**:
   ```bash
   # Check compose file syntax
   docker-compose -f docker-compose.production.yml config
   
   # Validate environment variables
   docker-compose -f docker-compose.production.yml config | grep -i "environment"
   ```

### Kubernetes Deployment Issues

#### Pod Failures

**ImagePullBackOff**:
```bash
# Check image name and tag
kubectl describe pod -l app=lightrag -n lightrag-production

# Verify image exists
docker pull ghcr.io/hkuds/lightrag:v1.2.3

# Check image pull secrets
kubectl get secrets -n lightrag-production
kubectl describe secret regcred -n lightrag-production
```

**CrashLoopBackOff**:
```bash
# Check pod logs
kubectl logs -l app=lightrag -n lightrag-production --previous

# Check resource limits
kubectl describe pod -l app=lightrag -n lightrag-production | grep -A 5 "Limits"

# Check liveness/readiness probes
kubectl describe pod -l app=lightrag -n lightrag-production | grep -A 10 "Liveness\|Readiness"
```

**Pending Pods**:
```bash
# Check node resources
kubectl describe nodes

# Check PVC status
kubectl get pvc -n lightrag-production

# Check pod scheduling constraints
kubectl describe pod -l app=lightrag -n lightrag-production | grep -A 5 "Events"
```

#### Service Discovery Issues

**Service Connectivity**:
```bash
# Test service resolution
kubectl run debug --rm -it --image=busybox -- nslookup lightrag.lightrag-production.svc.cluster.local

# Check service endpoints
kubectl get endpoints lightrag -n lightrag-production

# Test port connectivity
kubectl run debug --rm -it --image=busybox -- telnet lightrag.lightrag-production.svc.cluster.local 9621
```

#### Debug Commands

```bash
# Get all resources
kubectl get all -n lightrag-production

# Check events
kubectl get events -n lightrag-production --sort-by=.metadata.creationTimestamp

# Pod debugging
kubectl exec -it deployment/lightrag -n lightrag-production -- bash

# Port forward for testing
kubectl port-forward svc/lightrag 8080:9621 -n lightrag-production
```

## Container Issues

### Runtime Problems

#### Application Startup Failures

**Symptoms**:
- Container exits immediately
- Health checks failing
- Port binding errors

**Debug Steps**:
```bash
# Check container logs
docker logs lightrag_app_1 --details

# Run container interactively
docker run -it --rm lightrag:latest bash

# Check process status
docker exec lightrag_app_1 ps aux

# Verify port binding
docker exec lightrag_app_1 netstat -tlnp
```

#### Resource Issues

**Memory Problems**:
```bash
# Check memory usage
docker stats lightrag_app_1

# Monitor memory over time
while true; do
  docker exec lightrag_app_1 free -m
  sleep 5
done

# Check for memory leaks
docker exec lightrag_app_1 cat /proc/meminfo
```

**CPU Issues**:
```bash
# Monitor CPU usage
docker stats --no-stream lightrag_app_1

# Check CPU-intensive processes
docker exec lightrag_app_1 top

# Profile application
docker exec lightrag_app_1 python -m cProfile -s cumulative -m lightrag.api.lightrag_server
```

#### Network Problems

**Port Accessibility**:
```bash
# Check port mapping
docker port lightrag_app_1

# Test connectivity
curl -f http://localhost:9621/health

# Check container network
docker network ls
docker network inspect bridge
```

**DNS Resolution**:
```bash
# Test DNS inside container
docker exec lightrag_app_1 nslookup postgres
docker exec lightrag_app_1 cat /etc/resolv.conf

# Check hosts file
docker exec lightrag_app_1 cat /etc/hosts
```

### Container Build Issues

#### Dockerfile Problems

**Layer Caching Issues**:
```bash
# Build without cache
docker build --no-cache -t lightrag:debug .

# Check layer sizes
docker history lightrag:latest

# Optimize layer ordering
# Move frequently changing files (code) to end
# Keep stable dependencies at beginning
```

**Dependency Installation**:
```bash
# Debug pip installation
docker run --rm python:3.10-slim pip install --verbose lightrag-hku

# Check package conflicts
docker run --rm python:3.10-slim pip check

# Use specific package versions
pip install "numpy==1.24.0" "pandas>=2.0.0"
```

## Database Problems

### Connection Issues

#### PostgreSQL Problems

**Connection Refused**:
```bash
# Check PostgreSQL status
docker exec postgres_container pg_isready

# Verify connection parameters
psql "postgresql://lightrag:password@localhost:5432/lightrag" -c "SELECT version();"

# Check logs
docker logs postgres_container --tail=50
```

**Authentication Failures**:
```bash
# Verify credentials
echo "Host: $POSTGRES_HOST"
echo "User: $POSTGRES_USER"
echo "Database: $POSTGRES_DB"

# Test authentication
PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT current_user;"
```

**Extension Issues**:
```bash
# Check vector extension
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT * FROM pg_extension WHERE extname='vector';"

# Install missing extensions
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### Redis Problems

**Connection Issues**:
```bash
# Test Redis connectivity
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping

# Check Redis logs
docker logs redis_container --tail=50

# Verify Redis configuration
redis-cli -h $REDIS_HOST -p $REDIS_PORT CONFIG GET "*"
```

**Memory Issues**:
```bash
# Check Redis memory usage
redis-cli -h $REDIS_HOST -p $REDIS_PORT INFO memory

# Monitor Redis performance
redis-cli -h $REDIS_HOST -p $REDIS_PORT --latency

# Clear Redis data if needed
redis-cli -h $REDIS_HOST -p $REDIS_PORT FLUSHALL
```

### Data Issues

#### Migration Problems

**Schema Mismatches**:
```bash
# Check table structure
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "\d+ documents"

# Compare schemas
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB --schema-only | grep CREATE

# Manual migration
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < migration.sql
```

**Data Corruption**:
```bash
# Check data integrity
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT COUNT(*) FROM documents WHERE content IS NULL;"

# Backup before repair
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB > backup_$(date +%Y%m%d).sql

# Restore from backup
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB < backup_20241201.sql
```

## Network and Connectivity

### Service Discovery

#### Internal Communication

**Service-to-Service Communication**:
```bash
# Test internal connectivity
docker exec lightrag_app_1 curl -f http://postgres:5432
docker exec lightrag_app_1 curl -f http://redis:6379

# Check docker network
docker network inspect lightrag_default
```

**DNS Resolution**:
```bash
# Test DNS resolution
docker exec lightrag_app_1 nslookup postgres
docker exec lightrag_app_1 dig postgres

# Check DNS configuration
docker exec lightrag_app_1 cat /etc/resolv.conf
```

#### External Access

**Load Balancer Issues**:
```bash
# Check ingress status
kubectl get ingress -n lightrag-production

# Test external URL
curl -I https://lightrag.example.com/health

# Check SSL certificate
openssl s_client -connect lightrag.example.com:443 -servername lightrag.example.com
```

**Firewall Problems**:
```bash
# Check iptables rules
sudo iptables -L

# Test port accessibility
telnet lightrag.example.com 443
nc -zv lightrag.example.com 443
```

### Proxy and Gateway Issues

#### Nginx Configuration

**Proxy Problems**:
```bash
# Check nginx status
docker exec nginx_container nginx -t

# Test upstream connectivity
docker exec nginx_container curl -f http://lightrag:9621/health

# Check nginx logs
docker logs nginx_container --tail=50
```

**SSL Issues**:
```bash
# Verify certificate
docker exec nginx_container openssl x509 -in /etc/ssl/certs/lightrag.crt -text -noout

# Check certificate chain
docker exec nginx_container openssl verify -CAfile /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/lightrag.crt
```

## Performance Issues

### Response Time Problems

#### Slow Queries

**Database Performance**:
```bash
# Check slow queries
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC LIMIT 10;"

# Analyze query plans
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "
EXPLAIN ANALYZE SELECT * FROM documents WHERE content ILIKE '%search%';"
```

**Application Profiling**:
```bash
# Profile Python application
python -m cProfile -s cumulative scripts/profile_app.py

# Memory profiling
python -m memory_profiler scripts/profile_memory.py

# Line profiling
kernprof -l -v scripts/profile_lines.py
```

#### Resource Constraints

**CPU Bottlenecks**:
```bash
# Monitor CPU usage
top -p $(pgrep -f lightrag)

# Check CPU limits
docker exec lightrag_app_1 cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us

# Profile CPU usage
perf top -p $(pgrep -f lightrag)
```

**Memory Issues**:
```bash
# Monitor memory usage
free -h
ps aux | grep lightrag | awk '{print $4, $11}'

# Check memory limits
docker exec lightrag_app_1 cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# Memory leak detection
valgrind --leak-check=full python -m lightrag.api.lightrag_server
```

### Scalability Problems

#### Horizontal Scaling

**Load Distribution**:
```bash
# Check pod distribution
kubectl get pods -n lightrag-production -o wide

# Monitor request distribution
kubectl top pods -n lightrag-production

# Check HPA status
kubectl get hpa -n lightrag-production
```

**Session Affinity**:
```bash
# Check service configuration
kubectl describe service lightrag -n lightrag-production

# Test session consistency
for i in {1..10}; do
  curl -b cookies.txt -c cookies.txt https://lightrag.example.com/api/session
done
```

## Security and Authentication

### Authentication Issues

#### JWT Token Problems

**Token Validation**:
```bash
# Decode JWT token
echo "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." | base64 -d

# Verify token signature
python -c "
import jwt
token = 'your_jwt_token_here'
secret = 'your_secret_key'
try:
    payload = jwt.decode(token, secret, algorithms=['HS256'])
    print('Valid token:', payload)
except jwt.InvalidTokenError as e:
    print('Invalid token:', e)
"
```

**Session Management**:
```bash
# Check Redis sessions
redis-cli -h $REDIS_HOST KEYS "session:*"

# Clear expired sessions
redis-cli -h $REDIS_HOST EVAL "
local keys = redis.call('keys', 'session:*')
for i=1,#keys do
  local ttl = redis.call('ttl', keys[i])
  if ttl == -1 then
    redis.call('del', keys[i])
  end
end
return #keys
" 0
```

#### API Key Issues

**Key Validation**:
```bash
# Test API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://lightrag.example.com/api/health

# Check key permissions
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://lightrag.example.com/api/user/profile
```

**Rate Limiting**:
```bash
# Check rate limit headers
curl -I -H "Authorization: Bearer YOUR_API_KEY" \
  https://lightrag.example.com/api/query

# Monitor rate limiting
redis-cli -h $REDIS_HOST MONITOR | grep "rate_limit"
```

### SSL/TLS Issues

#### Certificate Problems

**Certificate Validation**:
```bash
# Check certificate expiration
openssl x509 -in certificate.crt -noout -dates

# Verify certificate chain
openssl verify -CAfile ca-bundle.crt certificate.crt

# Test SSL connection
openssl s_client -connect lightrag.example.com:443 -verify_return_error
```

**Certificate Renewal**:
```bash
# Check cert-manager status (Kubernetes)
kubectl get certificates -n lightrag-production
kubectl describe certificate lightrag-tls -n lightrag-production

# Manual certificate renewal
certbot renew --dry-run
```

## Monitoring and Logging

### Log Analysis

#### Application Logs

**Log Aggregation Issues**:
```bash
# Check log rotation
ls -la /var/log/lightrag/

# Monitor log growth
du -sh /var/log/lightrag/app.log

# Parse JSON logs
tail -f /var/log/lightrag/app.log | jq '.level, .message'
```

**Error Pattern Analysis**:
```bash
# Find error patterns
grep -E "(ERROR|CRITICAL)" /var/log/lightrag/app.log | \
  awk '{print $NF}' | sort | uniq -c | sort -nr

# Check for memory leaks
grep "OutOfMemoryError" /var/log/lightrag/app.log

# Monitor API errors
grep "HTTP 5" /var/log/lightrag/access.log | wc -l
```

#### System Logs

**Container Logs**:
```bash
# Aggregate container logs
docker logs lightrag_app_1 2>&1 | tee lightrag_debug.log

# Follow logs in real-time
docker logs -f lightrag_app_1 | grep -E "(ERROR|WARNING)"

# Export logs for analysis
docker logs lightrag_app_1 > logs_$(date +%Y%m%d_%H%M%S).txt
```

**Kubernetes Logs**:
```bash
# Aggregate pod logs
kubectl logs -l app=lightrag -n lightrag-production --all-containers=true

# Export logs
kubectl logs -l app=lightrag -n lightrag-production > k8s_logs.txt

# Stream logs
kubectl logs -f deployment/lightrag -n lightrag-production
```

### Metrics Collection

#### Prometheus Issues

**Metric Collection Problems**:
```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Verify metric exposure
curl http://lightrag:9621/metrics

# Test metric queries
curl 'http://prometheus:9090/api/v1/query?query=lightrag_requests_total'
```

**Dashboard Issues**:
```bash
# Check Grafana connectivity
curl http://grafana:3000/api/health

# Verify data source
curl http://grafana:3000/api/datasources

# Test dashboard queries
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
  http://grafana:3000/api/dashboards/uid/lightrag-dashboard
```

## Debug Tools and Commands

### Quick Diagnostic Commands

#### System Status Check
```bash
#!/bin/bash
# Quick system status check

echo "=== LightRAG System Status ==="
echo "Date: $(date)"
echo

echo "=== Application Status ==="
curl -s http://localhost:9621/health | jq . || echo "Application not responding"
echo

echo "=== Database Status ==="
pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT && echo "PostgreSQL: OK" || echo "PostgreSQL: FAILED"
redis-cli -h $REDIS_HOST ping | grep -q PONG && echo "Redis: OK" || echo "Redis: FAILED"
echo

echo "=== Container Status ==="
docker ps --filter "name=lightrag" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo

echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

#### Log Analysis Script
```bash
#!/bin/bash
# Analyze recent logs for issues

LOG_FILE="/var/log/lightrag/app.log"
HOURS_BACK=2

echo "=== Log Analysis (Last $HOURS_BACK hours) ==="
echo

# Find errors
echo "Recent Errors:"
grep -E "(ERROR|CRITICAL)" $LOG_FILE | \
  tail -n 50 | \
  awk '{print substr($0, 1, 100)"..."}' | \
  sort | uniq -c | sort -nr
echo

# Find warnings
echo "Recent Warnings:"
grep "WARNING" $LOG_FILE | \
  tail -n 20 | \
  awk '{print substr($0, 1, 100)"..."}' | \
  sort | uniq -c | sort -nr
echo

# Performance issues
echo "Slow Requests (>1s):"
grep "request_duration" $LOG_FILE | \
  awk '$NF > 1000 {print $0}' | \
  tail -n 10
```

#### Health Check Script
```bash
#!/bin/bash
# Comprehensive health check

set -e

echo "=== Comprehensive Health Check ==="

# Basic connectivity
echo "Checking basic connectivity..."
curl -f http://localhost:9621/health >/dev/null && echo "✓ Basic health OK" || echo "✗ Basic health FAILED"

# Detailed health
echo "Checking detailed health..."
HEALTH_RESPONSE=$(curl -s http://localhost:9621/api/health)
echo "$HEALTH_RESPONSE" | jq -e '.status == "healthy"' >/dev/null && echo "✓ Detailed health OK" || echo "✗ Detailed health FAILED"

# Database connectivity
echo "Checking database connectivity..."
pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT >/dev/null && echo "✓ PostgreSQL OK" || echo "✗ PostgreSQL FAILED"
redis-cli -h $REDIS_HOST ping >/dev/null && echo "✓ Redis OK" || echo "✗ Redis FAILED"

# API functionality
echo "Checking API functionality..."
curl -s -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "mode": "naive"}' | \
  jq -e 'has("response")' >/dev/null && echo "✓ API functionality OK" || echo "✗ API functionality FAILED"

echo "Health check completed."
```

### Performance Profiling

#### Application Profiling Script
```bash
#!/bin/bash
# Profile application performance

PID=$(pgrep -f "lightrag.api.lightrag_server")

if [ -z "$PID" ]; then
  echo "LightRAG process not found"
  exit 1
fi

echo "Profiling LightRAG process (PID: $PID)"

# CPU profiling
echo "CPU usage over 60 seconds:"
for i in {1..12}; do
  CPU=$(ps -p $PID -o %cpu --no-headers)
  echo "$(date +%H:%M:%S): $CPU%"
  sleep 5
done

# Memory profiling
echo "Memory usage:"
ps -p $PID -o pid,ppid,%mem,vsz,rss,comm --no-headers

# Network connections
echo "Network connections:"
netstat -tlnp | grep $PID
```

### Automated Recovery Scripts

#### Service Recovery Script
```bash
#!/bin/bash
# Automatic service recovery

SERVICE_URL="http://localhost:9621/health"
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if curl -f $SERVICE_URL >/dev/null 2>&1; then
    echo "Service is healthy"
    exit 0
  fi
  
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "Service unhealthy, attempt $RETRY_COUNT/$MAX_RETRIES"
  
  if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
    echo "Restarting service..."
    docker-compose restart lightrag
    sleep 30
  fi
done

echo "Service recovery failed after $MAX_RETRIES attempts"
echo "Manual intervention required"
exit 1
```

---

This troubleshooting guide covers the most common issues encountered in LightRAG CI/CD operations. For complex issues not covered here, escalate to the development team with detailed logs and error descriptions.