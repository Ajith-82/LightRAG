# Deployment Runbook for LightRAG

This runbook provides step-by-step procedures for deploying LightRAG across different environments.

## Table of Contents

- [Pre-deployment Checklist](#pre-deployment-checklist)
- [Environment Setup](#environment-setup)
- [Staging Deployment](#staging-deployment)
- [Production Deployment](#production-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Rollback Procedures](#rollback-procedures)
- [Post-deployment Validation](#post-deployment-validation)
- [Emergency Procedures](#emergency-procedures)
- [Maintenance Windows](#maintenance-windows)

## Pre-deployment Checklist

### Code Readiness

- [ ] All tests passing in CI/CD pipeline
- [ ] Code coverage meets threshold (≥70%)
- [ ] Security scans completed without critical issues
- [ ] Performance benchmarks within acceptable range
- [ ] Documentation updated for new features

### Infrastructure Readiness

- [ ] Target environment health verified
- [ ] Database migrations reviewed and tested
- [ ] Environment variables configured
- [ ] Secrets and credentials rotated if needed
- [ ] Backup systems verified and functional

### Team Coordination

- [ ] Deployment window scheduled and communicated
- [ ] On-call engineer identified and available
- [ ] Rollback plan reviewed and understood
- [ ] Stakeholders notified of deployment

### Release Information

- [ ] Release notes prepared
- [ ] Version number confirmed
- [ ] Change log updated
- [ ] Breaking changes documented
- [ ] Migration guides provided if needed

## Environment Setup

### Development Environment

**Purpose**: Local development and testing

**Setup**:
```bash
# Clone repository
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# Set up development environment
make dev-setup

# Start services
make db-setup
make dev-server
```

**Validation**:
```bash
# Run health check
curl http://localhost:9621/health

# Run basic tests
make dev-test
```

### Staging Environment

**Purpose**: Pre-production validation and integration testing

**Requirements**:
- Production-like configuration
- Real database with test data
- External service integrations
- Performance monitoring

**Configuration**:
```bash
# Environment variables
NODE_ENV=staging
DEBUG=false
LOG_LEVEL=DEBUG
AUTH_ENABLED=true
RATE_LIMIT_ENABLED=true
```

### Production Environment

**Purpose**: Live user-facing deployment

**Requirements**:
- High availability configuration
- Encrypted data storage
- Comprehensive monitoring
- Backup and disaster recovery

**Configuration**:
```bash
# Environment variables
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO
AUTH_ENABLED=true
RATE_LIMIT_ENABLED=true
JWT_EXPIRE_HOURS=24
```

## Staging Deployment

### Automatic Deployment (Main Branch)

Staging deployments are triggered automatically when code is merged to the main branch.

**Process**:
1. Code merged to main branch
2. CI pipeline executes automatically
3. Docker images built and pushed
4. Staging deployment initiated
5. Health checks performed
6. Notification sent to team

**Monitoring**:
```bash
# Check GitHub Actions status
# Visit: https://github.com/HKUDS/LightRAG/actions

# Monitor deployment progress
./scripts/deploy/health-check.sh -e staging

# View logs
make logs ENVIRONMENT=staging
```

### Manual Staging Deployment

For testing specific versions or configurations:

**Steps**:
1. **Prepare Environment**
   ```bash
   export ENVIRONMENT=staging
   export IMAGE_TAG=v1.2.3  # or specific tag
   ```

2. **Deploy Application**
   ```bash
   ./scripts/deploy/deploy-docker.sh -e staging -t $IMAGE_TAG
   ```

3. **Verify Deployment**
   ```bash
   ./scripts/deploy/health-check.sh -e staging --comprehensive
   ```

4. **Run Integration Tests**
   ```bash
   ./scripts/ci/integration-tests.sh -e staging
   ```

### Staging Validation Checklist

- [ ] Application starts successfully
- [ ] Health endpoints respond correctly
- [ ] Database connectivity verified
- [ ] API endpoints functional
- [ ] Authentication working
- [ ] File upload/processing working
- [ ] Query functionality operational
- [ ] Performance within acceptable range

## Production Deployment

### Release Deployment (Tagged Versions)

Production deployments are triggered by creating version tags.

**Process**:
1. **Create Release Tag**
   ```bash
   # Tag the release
   git tag -a v1.2.3 -m "Release version 1.2.3"
   git push origin v1.2.3
   ```

2. **Monitor Release Pipeline**
   - GitHub Actions will build and test the release
   - Docker images will be created with version tags
   - Production deployment will require manual approval

3. **Approve Production Deployment**
   - Review release notes and test results
   - Approve deployment in GitHub Actions
   - Monitor deployment progress

### Manual Production Deployment

For emergency deployments or specific scenarios:

**Prerequisites**:
- Emergency change approval (if applicable)
- Production deployment window
- On-call engineer available

**Steps**:
1. **Preparation**
   ```bash
   export ENVIRONMENT=production
   export IMAGE_TAG=v1.2.3
   ```

2. **Pre-deployment Backup**
   ```bash
   ./backup/backup-script.sh
   ```

3. **Deploy Application**
   ```bash
   ./scripts/deploy/deploy-docker.sh -e production -t $IMAGE_TAG
   ```

4. **Verify Deployment**
   ```bash
   ./scripts/deploy/health-check.sh -e production --comprehensive
   ```

5. **Run Smoke Tests**
   ```bash
   # Basic functionality verification
   curl -f $PRODUCTION_URL/health
   curl -f $PRODUCTION_URL/api/health
   ```

### Production Deployment Checklist

- [ ] Backup completed successfully
- [ ] Application deployed without errors
- [ ] Health checks passing
- [ ] Database migrations applied (if any)
- [ ] SSL certificates valid
- [ ] CDN/Cache cleared (if applicable)
- [ ] Monitoring alerts functional
- [ ] Performance metrics normal
- [ ] User authentication working
- [ ] Core functionality operational

## Kubernetes Deployment

### Prerequisites

**Cluster Requirements**:
- Kubernetes 1.24+
- Ingress controller (nginx recommended)
- cert-manager for SSL certificates
- Persistent volume provisioner

**Access Requirements**:
```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes

# Check namespace
kubectl get namespace lightrag-production || \
kubectl create namespace lightrag-production
```

### Helm Deployment (Recommended)

**Steps**:
1. **Update Helm Dependencies**
   ```bash
   cd k8s-deploy/lightrag
   helm dependency update
   ```

2. **Deploy with Helm**
   ```bash
   ./scripts/deploy/deploy-k8s.sh -e production \
     -t v1.2.3 -n lightrag-production
   ```

3. **Verify Deployment**
   ```bash
   # Check deployment status
   kubectl rollout status deployment/lightrag -n lightrag-production
   
   # Check pods
   kubectl get pods -n lightrag-production -l app=lightrag
   
   # Check services
   kubectl get services -n lightrag-production
   ```

### kubectl Deployment (Alternative)

If Helm is not available:

**Steps**:
1. **Generate Manifests**
   ```bash
   ./scripts/deploy/deploy-k8s.sh -e production \
     -t v1.2.3 -n lightrag-production
   ```

2. **Review Manifests**
   ```bash
   ls k8s-manifests/
   cat k8s-manifests/deployment.yaml
   ```

3. **Apply Manifests**
   ```bash
   kubectl apply -f k8s-manifests/ -n lightrag-production
   ```

### Kubernetes Validation Checklist

- [ ] Pods running and ready
- [ ] Services exposing correct ports
- [ ] Ingress configured (if applicable)
- [ ] Persistent volumes mounted
- [ ] ConfigMaps and Secrets applied
- [ ] Health checks passing
- [ ] Resource limits appropriate
- [ ] Network policies applied (if used)

## Rollback Procedures

### Automatic Rollback

The deployment scripts include automatic rollback on failure:

```bash
# Automatic rollback is enabled by default
ROLLBACK_ON_FAILURE=true ./scripts/deploy/deploy-docker.sh
```

### Manual Rollback

#### Docker Rollback

**Steps**:
1. **Identify Current Version**
   ```bash
   docker ps --filter "name=lightrag" \
     --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
   ```

2. **Execute Rollback**
   ```bash
   ./scripts/deploy/rollback.sh -e production -t docker
   ```

3. **Verify Rollback**
   ```bash
   ./scripts/deploy/health-check.sh -e production
   ```

#### Kubernetes Rollback

**Steps**:
1. **Check Rollout History**
   ```bash
   kubectl rollout history deployment/lightrag -n lightrag-production
   ```

2. **Execute Rollback**
   ```bash
   ./scripts/deploy/rollback.sh -e production -t kubernetes
   ```

3. **Verify Rollback**
   ```bash
   kubectl rollout status deployment/lightrag -n lightrag-production
   ```

#### Helm Rollback

**Steps**:
1. **Check Release History**
   ```bash
   helm history lightrag -n lightrag-production
   ```

2. **Execute Rollback**
   ```bash
   helm rollback lightrag -n lightrag-production
   ```

### Emergency Rollback

For critical production issues:

**Immediate Actions**:
1. **Stop Traffic** (if load balancer available)
   ```bash
   # Remove from load balancer or scale to 0
   kubectl scale deployment/lightrag --replicas=0 -n lightrag-production
   ```

2. **Execute Fast Rollback**
   ```bash
   ./scripts/deploy/rollback.sh -e production --no-confirm
   ```

3. **Verify Service Restoration**
   ```bash
   ./scripts/deploy/health-check.sh -e production
   ```

4. **Restore Traffic**
   ```bash
   kubectl scale deployment/lightrag --replicas=3 -n lightrag-production
   ```

### Rollback Checklist

- [ ] Rollback reason documented
- [ ] Previous version identified
- [ ] Rollback executed successfully
- [ ] Health checks passing
- [ ] User functionality restored
- [ ] Monitoring alerts cleared
- [ ] Incident report created
- [ ] Team notified of resolution

## Post-deployment Validation

### Health Check Validation

**Basic Health Checks**:
```bash
# Application health
curl -f $BASE_URL/health

# Detailed health
curl -f $BASE_URL/api/health | jq .
```

**Comprehensive Validation**:
```bash
./scripts/deploy/health-check.sh -e $ENVIRONMENT --comprehensive
```

### Functional Testing

**Core Functionality**:
```bash
# Document upload
curl -X POST $BASE_URL/documents/insert \
  -H "Content-Type: application/json" \
  -d '{"text": "Test document", "description": "Deployment validation"}'

# Query functionality
curl -X POST $BASE_URL/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "mode": "naive"}'
```

**Integration Testing**:
```bash
# Run integration test suite
./scripts/ci/integration-tests.sh -e $ENVIRONMENT
```

### Performance Validation

**Response Time Check**:
```bash
# Measure response times
time curl -s $BASE_URL/health

# Load testing (if needed)
make test-performance
```

**Resource Monitoring**:
```bash
# Docker resource usage
docker stats lightrag_app_1

# Kubernetes resource usage
kubectl top pods -n lightrag-$ENVIRONMENT
```

### Monitoring Verification

**Check Dashboards**:
- Application performance metrics
- Infrastructure resource usage
- Error rates and alerts
- User activity patterns

**Verify Alerts**:
- Health check monitoring active
- Performance threshold alerts configured
- Error rate monitoring functional
- Security alert systems operational

## Emergency Procedures

### Service Outage Response

**Immediate Response (0-5 minutes)**:
1. **Assess Impact**
   ```bash
   ./scripts/deploy/health-check.sh -e production
   make status ENVIRONMENT=production
   ```

2. **Check Recent Changes**
   ```bash
   git log --oneline -10
   kubectl rollout history deployment/lightrag -n lightrag-production
   ```

3. **Initiate Rollback** (if deployment-related)
   ```bash
   ./scripts/deploy/rollback.sh -e production --no-confirm
   ```

**Short-term Response (5-15 minutes)**:
1. **Investigate Root Cause**
   ```bash
   make logs ENVIRONMENT=production
   kubectl describe pods -l app=lightrag -n lightrag-production
   ```

2. **Implement Workaround** (if possible)
   - Scale resources
   - Restart services
   - Clear caches

3. **Communicate Status**
   - Update status page
   - Notify stakeholders
   - Document timeline

**Long-term Response (15+ minutes)**:
1. **Implement Permanent Fix**
2. **Validate Solution**
3. **Conduct Post-incident Review**
4. **Update Procedures**

### Security Incident Response

**Immediate Actions**:
1. **Isolate Affected Systems**
   ```bash
   # Scale down to stop traffic
   kubectl scale deployment/lightrag --replicas=0 -n lightrag-production
   ```

2. **Assess Scope**
   - Check logs for suspicious activity
   - Review recent access patterns
   - Validate data integrity

3. **Secure Environment**
   - Rotate credentials
   - Update firewall rules
   - Apply security patches

**Follow-up Actions**:
1. **Investigate Incident**
2. **Implement Security Improvements**
3. **Update Security Procedures**
4. **Conduct Security Review**

### Data Loss/Corruption Response

**Immediate Actions**:
1. **Stop Write Operations**
   ```bash
   # Scale application to read-only if possible
   kubectl annotate deployment/lightrag maintenance.mode=read-only
   ```

2. **Assess Data Impact**
   - Identify affected data
   - Determine scope of corruption
   - Check backup integrity

3. **Initiate Data Recovery**
   ```bash
   ./backup/restore-script.sh
   ```

**Recovery Process**:
1. **Restore from Backup**
2. **Validate Data Integrity**
3. **Resume Normal Operations**
4. **Conduct Data Recovery Review**

## Maintenance Windows

### Scheduled Maintenance

**Planning Phase**:
- [ ] Maintenance window scheduled
- [ ] Stakeholders notified
- [ ] Rollback plan prepared
- [ ] Team assignments confirmed

**Pre-maintenance Checklist**:
- [ ] Backup completed
- [ ] Maintenance banner displayed
- [ ] Monitoring alerts acknowledged
- [ ] Emergency contacts available

**Maintenance Execution**:
1. **Begin Maintenance Window**
   ```bash
   # Display maintenance mode
   kubectl annotate deployment/lightrag maintenance.active=true
   ```

2. **Perform Updates**
   - Apply security patches
   - Update dependencies
   - Migrate data if needed
   - Update configurations

3. **Validate Changes**
   ```bash
   ./scripts/deploy/health-check.sh -e production --comprehensive
   ```

4. **End Maintenance Window**
   ```bash
   # Remove maintenance mode
   kubectl annotate deployment/lightrag maintenance.active-
   ```

**Post-maintenance Checklist**:
- [ ] All services operational
- [ ] Performance metrics normal
- [ ] User functionality verified
- [ ] Monitoring alerts functional
- [ ] Maintenance report completed

### Emergency Maintenance

For critical security patches or urgent fixes:

**Authorization**:
- Emergency change approval
- Stakeholder notification
- Risk assessment documented

**Execution**:
- Follow abbreviated maintenance process
- Prioritize safety over convenience
- Document all changes thoroughly

**Communication**:
- Immediate notification to stakeholders
- Regular status updates
- Post-maintenance summary

---

This runbook provides comprehensive procedures for LightRAG deployments. Always follow the appropriate checklist and escalate to senior team members when in doubt.