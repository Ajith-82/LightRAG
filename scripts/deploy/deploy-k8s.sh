#!/bin/bash
set -euo pipefail

# Kubernetes deployment script
# This script automates Kubernetes deployments for LightRAG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration with defaults
ENVIRONMENT=${ENVIRONMENT:-staging}
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-ghcr.io}
IMAGE_NAME=${IMAGE_NAME:-hkuds/lightrag}
NAMESPACE=${NAMESPACE:-lightrag-${ENVIRONMENT}}
KUBECONFIG=${KUBECONFIG:-~/.kube/config}
HELM_RELEASE=${HELM_RELEASE:-lightrag}
DEPLOYMENT_TIMEOUT=${DEPLOYMENT_TIMEOUT:-600}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking Kubernetes deployment dependencies..."

    local missing_deps=()

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        missing_deps+=("kubectl")
    fi

    # Check helm (optional but recommended)
    if ! command -v helm &> /dev/null; then
        log_warning "Helm not found - will use kubectl manifests"
    fi

    # Check curl for health checks
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    # Check jq for JSON processing
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found - JSON parsing may be limited"
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi

    log_success "Dependencies check completed"
}

validate_kubernetes_access() {
    log_info "Validating Kubernetes cluster access..."

    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_info "Check your kubeconfig: $KUBECONFIG"
        exit 1
    fi

    # Check if namespace exists, create if not
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
    fi

    # Check RBAC permissions
    if ! kubectl auth can-i create deployments -n "$NAMESPACE" &> /dev/null; then
        log_error "Insufficient RBAC permissions for namespace: $NAMESPACE"
        exit 1
    fi

    log_success "Kubernetes access validated"
}

prepare_kubernetes_manifests() {
    log_info "Preparing Kubernetes manifests..."

    local manifests_dir="k8s-manifests"
    mkdir -p "$manifests_dir"

    local full_image_name="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    # Create ConfigMap for environment variables
    cat > "$manifests_dir/configmap.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: lightrag-config
  namespace: $NAMESPACE
data:
  NODE_ENV: "$ENVIRONMENT"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  PORT: "9621"
  HOST: "0.0.0.0"
  POSTGRES_HOST: "postgres"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "lightrag"
  REDIS_URL: "redis://redis:6379"
EOF

    # Create Secret for sensitive data
    cat > "$manifests_dir/secret.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: lightrag-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  # Base64 encoded values - update these with actual secrets
  postgres-password: $(echo -n "lightrag" | base64)
  jwt-secret-key: $(echo -n "$(openssl rand -base64 32)" | base64)
EOF

    # Create Deployment
    cat > "$manifests_dir/deployment.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag
  namespace: $NAMESPACE
  labels:
    app: lightrag
    version: "$IMAGE_TAG"
    environment: "$ENVIRONMENT"
spec:
  replicas: $(case $ENVIRONMENT in production) echo 3;; staging) echo 2;; *) echo 1;; esac)
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: lightrag
  template:
    metadata:
      labels:
        app: lightrag
        version: "$IMAGE_TAG"
        environment: "$ENVIRONMENT"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: lightrag
        image: $full_image_name
        imagePullPolicy: Always
        ports:
        - containerPort: 9621
          name: http
          protocol: TCP
        env:
        - name: POSTGRES_USER
          value: "lightrag"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: lightrag-secrets
              key: postgres-password
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: lightrag-secrets
              key: jwt-secret-key
        envFrom:
        - configMapRef:
            name: lightrag-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9621
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 9621
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: data-storage
          mountPath: /app/rag_storage
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: lightrag-data
      restartPolicy: Always
EOF

    # Create Service
    cat > "$manifests_dir/service.yaml" << EOF
apiVersion: v1
kind: Service
metadata:
  name: lightrag
  namespace: $NAMESPACE
  labels:
    app: lightrag
spec:
  type: ClusterIP
  ports:
  - port: 9621
    targetPort: 9621
    protocol: TCP
    name: http
  selector:
    app: lightrag
EOF

    # Create Ingress (if in production)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        cat > "$manifests_dir/ingress.yaml" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lightrag
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
  - hosts:
    - lightrag.${DOMAIN:-example.com}
    secretName: lightrag-tls
  rules:
  - host: lightrag.${DOMAIN:-example.com}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: lightrag
            port:
              number: 9621
EOF
    fi

    # Create PersistentVolumeClaim
    cat > "$manifests_dir/pvc.yaml" << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: lightrag-data
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: $(case $ENVIRONMENT in production) echo "50Gi";; staging) echo "20Gi";; *) echo "10Gi";; esac)
  storageClassName: $(case $ENVIRONMENT in production) echo "fast-ssd";; *) echo "standard";; esac)
EOF

    # Create HorizontalPodAutoscaler (for production)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        cat > "$manifests_dir/hpa.yaml" << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lightrag
  namespace: $NAMESPACE
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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
    fi

    log_success "Kubernetes manifests prepared in $manifests_dir/"
}

deploy_with_helm() {
    log_info "Deploying with Helm..."

    local chart_dir="$PROJECT_ROOT/k8s-deploy/lightrag"

    if [[ ! -d "$chart_dir" ]]; then
        log_error "Helm chart not found at $chart_dir"
        return 1
    fi

    # Update Helm dependencies
    helm dependency update "$chart_dir"

    # Prepare values for this deployment
    local values_file="values-${ENVIRONMENT}.yaml"
    cat > "$values_file" << EOF
image:
  repository: $REGISTRY/$IMAGE_NAME
  tag: $IMAGE_TAG
  pullPolicy: Always

environment: $ENVIRONMENT

replicaCount: $(case $ENVIRONMENT in production) echo 3;; staging) echo 2;; *) echo 1;; esac)

resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

persistence:
  enabled: true
  size: $(case $ENVIRONMENT in production) echo "50Gi";; staging) echo "20Gi";; *) echo "10Gi";; esac)
  storageClass: $(case $ENVIRONMENT in production) echo "fast-ssd";; *) echo "standard";; esac)

autoscaling:
  enabled: $(case $ENVIRONMENT in production) echo "true";; *) echo "false";; esac)
  minReplicas: $(case $ENVIRONMENT in production) echo 3;; *) echo 1;; esac)
  maxReplicas: $(case $ENVIRONMENT in production) echo 10;; *) echo 3;; esac)

ingress:
  enabled: $(case $ENVIRONMENT in production) echo "true";; *) echo "false";; esac)
  hosts:
    - host: lightrag.${DOMAIN:-example.com}
      paths:
        - path: /
          pathType: Prefix

config:
  nodeEnv: $ENVIRONMENT
  debug: $(case $ENVIRONMENT in production) echo "false";; *) echo "true";; esac)
  logLevel: $(case $ENVIRONMENT in production) echo "INFO";; *) echo "DEBUG";; esac)
EOF

    # Deploy or upgrade
    if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
        log_info "Upgrading existing Helm release..."
        helm upgrade "$HELM_RELEASE" "$chart_dir" \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --timeout "${DEPLOYMENT_TIMEOUT}s" \
            --wait
    else
        log_info "Installing new Helm release..."
        helm install "$HELM_RELEASE" "$chart_dir" \
            --namespace "$NAMESPACE" \
            --values "$values_file" \
            --timeout "${DEPLOYMENT_TIMEOUT}s" \
            --wait
    fi

    log_success "Helm deployment completed"
}

deploy_with_kubectl() {
    log_info "Deploying with kubectl..."

    local manifests_dir="k8s-manifests"

    # Apply manifests in order
    local manifest_files=(
        "configmap.yaml"
        "secret.yaml"
        "pvc.yaml"
        "deployment.yaml"
        "service.yaml"
    )

    # Add optional manifests
    if [[ -f "$manifests_dir/ingress.yaml" ]]; then
        manifest_files+=("ingress.yaml")
    fi

    if [[ -f "$manifests_dir/hpa.yaml" ]]; then
        manifest_files+=("hpa.yaml")
    fi

    # Apply each manifest
    for manifest in "${manifest_files[@]}"; do
        log_info "Applying $manifest..."
        kubectl apply -f "$manifests_dir/$manifest"
    done

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/lightrag -n "$NAMESPACE" --timeout="${DEPLOYMENT_TIMEOUT}s"

    log_success "kubectl deployment completed"
}

verify_deployment() {
    log_info "Verifying deployment..."

    # Check pod status
    local pods=$(kubectl get pods -n "$NAMESPACE" -l app=lightrag --no-headers | wc -l)
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=lightrag --no-headers | grep "Running" | wc -l)

    log_info "Pods: $ready_pods/$pods ready"

    if [[ $ready_pods -eq 0 ]]; then
        log_error "No pods are ready"
        return 1
    fi

    # Check service endpoints
    local service_endpoints=$(kubectl get endpoints lightrag -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
    log_info "Service endpoints: $service_endpoints"

    # Get service details
    kubectl get service lightrag -n "$NAMESPACE"

    log_success "Deployment verification completed"
}

run_health_checks() {
    log_info "Running health checks..."

    # Port-forward to test the application
    log_info "Setting up port-forward for health checks..."
    kubectl port-forward -n "$NAMESPACE" service/lightrag 8080:9621 &
    PORT_FORWARD_PID=$!

    # Wait for port-forward to be ready
    sleep 5

    # Run health checks
    local health_check_passed=false
    local attempts=0
    local max_attempts=10

    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s -f "http://localhost:8080/health" >/dev/null 2>&1; then
            health_check_passed=true
            log_success "Health check passed"
            break
        else
            log_info "Health check attempt $((attempts + 1))/$max_attempts failed"
            sleep 10
            attempts=$((attempts + 1))
        fi
    done

    # Clean up port-forward
    kill $PORT_FORWARD_PID 2>/dev/null || true

    if [[ "$health_check_passed" == "false" ]]; then
        log_error "Health checks failed"
        return 1
    fi

    return 0
}

collect_deployment_info() {
    log_info "Collecting deployment information..."

    # Create deployment info
    cat > k8s-deployment-info.json << EOF
{
  "deployment": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "namespace": "$NAMESPACE",
    "image_tag": "$IMAGE_TAG",
    "image_name": "$REGISTRY/$IMAGE_NAME:$IMAGE_TAG",
    "helm_release": "$HELM_RELEASE",
    "deployed_by": "${USER:-unknown}",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')"
  },
  "resources": {}
}
EOF

    # Collect Kubernetes resource information
    if command -v jq &> /dev/null; then
        # Get pod information
        local pods_info=$(kubectl get pods -n "$NAMESPACE" -l app=lightrag -o json | \
            jq '.items[] | {name: .metadata.name, status: .status.phase, image: .spec.containers[0].image, node: .spec.nodeName}')

        # Get service information
        local service_info=$(kubectl get service lightrag -n "$NAMESPACE" -o json | \
            jq '{name: .metadata.name, type: .spec.type, clusterIP: .spec.clusterIP, ports: .spec.ports}')

        # Add to deployment info
        echo "$pods_info" | jq -s '.' | jq '. as $pods | {"pods": $pods}' > pods.json
        echo "$service_info" | jq '. as $svc | {"service": $svc}' > service.json

        jq -s '.[0] * .[1] * .[2]' k8s-deployment-info.json pods.json service.json > deployment-info.tmp && \
            mv deployment-info.tmp k8s-deployment-info.json

        rm -f pods.json service.json
    fi

    log_success "Deployment information collected: k8s-deployment-info.json"
}

rollback_deployment() {
    log_error "Deployment failed - initiating rollback..."

    if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
        log_info "Rolling back Helm release..."
        helm rollback "$HELM_RELEASE" -n "$NAMESPACE"
    else
        log_info "Rolling back kubectl deployment..."
        kubectl rollout undo deployment/lightrag -n "$NAMESPACE"
    fi

    # Wait for rollback to complete
    kubectl rollout status deployment/lightrag -n "$NAMESPACE" --timeout=300s || true

    log_warning "Rollback completed - check deployment status"
}

cleanup_deployment_files() {
    log_info "Cleaning up deployment files..."

    # Remove temporary files
    rm -rf k8s-manifests
    rm -f values-*.yaml

    log_success "Cleanup completed"
}

main() {
    log_info "Starting Kubernetes deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image: $REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    # Set up error handling
    if [[ "$ROLLBACK_ON_FAILURE" == "true" ]]; then
        trap 'rollback_deployment' ERR
    fi

    cd "$PROJECT_ROOT"

    check_dependencies
    validate_kubernetes_access

    # Choose deployment method
    if command -v helm &> /dev/null && [[ -d "k8s-deploy/lightrag" ]]; then
        deploy_with_helm
    else
        prepare_kubernetes_manifests
        deploy_with_kubectl
    fi

    verify_deployment

    if run_health_checks; then
        collect_deployment_info
        log_success "🚀 Kubernetes deployment completed successfully!"
    else
        log_error "Deployment health checks failed"
        exit 1
    fi

    cleanup_deployment_files

    # Print deployment summary
    log_info "=== Kubernetes Deployment Summary ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Pods: $(kubectl get pods -n "$NAMESPACE" -l app=lightrag --no-headers | grep "Running" | wc -l)"
    log_info "Service: $(kubectl get service lightrag -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')"
    log_info "Health Check: ✅"
    log_info "===================================="
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Deployment environment (staging, production)
    -t, --tag TAG             Docker image tag (default: latest)
    -r, --registry REGISTRY   Docker registry (default: ghcr.io)
    -i, --image IMAGE         Image name (default: hkuds/lightrag)
    -n, --namespace NAMESPACE Kubernetes namespace (default: lightrag-ENV)
    -R, --release RELEASE     Helm release name (default: lightrag)
    -T, --timeout SECONDS     Deployment timeout (default: 600)
    --no-rollback             Disable automatic rollback on failure
    -h, --help                Show this help message

Environment Variables:
    ENVIRONMENT              Deployment environment
    IMAGE_TAG               Docker image tag
    REGISTRY                Docker registry
    IMAGE_NAME              Docker image name
    NAMESPACE               Kubernetes namespace
    HELM_RELEASE            Helm release name
    DEPLOYMENT_TIMEOUT      Deployment timeout in seconds
    ROLLBACK_ON_FAILURE     Enable/disable rollback on failure
    KUBECONFIG              Path to kubeconfig file
    DOMAIN                  Domain for ingress (production only)

Examples:
    $0 -e staging -t v1.2.3                     # Deploy to staging
    $0 -e production -t latest -n lightrag-prod # Deploy to production
    ENVIRONMENT=staging IMAGE_TAG=dev $0         # Use environment variables
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -R|--release)
            HELM_RELEASE="$2"
            shift 2
            ;;
        -T|--timeout)
            DEPLOYMENT_TIMEOUT="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Update namespace based on environment if not explicitly set
if [[ "$NAMESPACE" == "lightrag-staging" ]] && [[ "$ENVIRONMENT" != "staging" ]]; then
    NAMESPACE="lightrag-${ENVIRONMENT}"
fi

# Run main function
main "$@"
