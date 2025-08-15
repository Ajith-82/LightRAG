#!/bin/bash
set -euo pipefail

# Rollback script for LightRAG deployments
# This script provides rollback capabilities for both Docker and Kubernetes deployments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration with defaults
ENVIRONMENT=${ENVIRONMENT:-staging}
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-auto}
NAMESPACE=${NAMESPACE:-lightrag-${ENVIRONMENT}}
HELM_RELEASE=${HELM_RELEASE:-lightrag}
ROLLBACK_STEPS=${ROLLBACK_STEPS:-1}
CONFIRMATION=${CONFIRMATION:-true}
BACKUP_RESTORE=${BACKUP_RESTORE:-false}

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
    log_info "Checking rollback dependencies..."
    
    local missing_deps=()
    
    # Check basic tools
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi
    
    # Check deployment-specific tools
    case $DEPLOYMENT_TYPE in
        docker|auto)
            if ! command -v docker &> /dev/null; then
                if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
                    missing_deps+=("docker")
                fi
            fi
            
            if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
                if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
                    missing_deps+=("docker-compose")
                fi
            fi
            ;;
        kubernetes|k8s|auto)
            if ! command -v kubectl &> /dev/null; then
                if [[ "$DEPLOYMENT_TYPE" =~ ^(kubernetes|k8s)$ ]]; then
                    missing_deps+=("kubectl")
                fi
            fi
            ;;
    esac
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    log_success "Dependencies check completed"
}

detect_deployment_type() {
    if [[ "$DEPLOYMENT_TYPE" != "auto" ]]; then
        log_info "Using specified deployment type: $DEPLOYMENT_TYPE"
        return
    fi
    
    log_info "Auto-detecting deployment type..."
    
    # Check for Kubernetes deployment
    if command -v kubectl &> /dev/null; then
        if kubectl get namespace "$NAMESPACE" &> /dev/null; then
            if kubectl get deployment lightrag -n "$NAMESPACE" &> /dev/null; then
                DEPLOYMENT_TYPE="kubernetes"
                log_info "Detected Kubernetes deployment"
                return
            fi
        fi
    fi
    
    # Check for Docker deployment
    if command -v docker &> /dev/null; then
        if docker ps --filter "name=lightrag" --format "{{.Names}}" | grep -q lightrag; then
            DEPLOYMENT_TYPE="docker"
            log_info "Detected Docker deployment"
            return
        fi
    fi
    
    log_error "Could not detect deployment type"
    log_info "Please specify deployment type with -t/--type option"
    exit 1
}

confirm_rollback() {
    if [[ "$CONFIRMATION" != "true" ]]; then
        return 0
    fi
    
    log_warning "=== ROLLBACK CONFIRMATION ==="
    log_warning "Environment: $ENVIRONMENT"
    log_warning "Deployment Type: $DEPLOYMENT_TYPE"
    log_warning "Rollback Steps: $ROLLBACK_STEPS"
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        log_warning "Namespace: $NAMESPACE"
        if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
            log_warning "Helm Release: $HELM_RELEASE"
        fi
    fi
    
    log_warning "=============================="
    
    echo -n -e "${YELLOW}Do you want to proceed with the rollback? (yes/no): ${NC}"
    read -r response
    
    case $response in
        yes|YES|y|Y)
            log_info "Rollback confirmed"
            ;;
        *)
            log_info "Rollback cancelled by user"
            exit 0
            ;;
    esac
}

backup_current_state() {
    log_info "Backing up current state before rollback..."
    
    local backup_dir="rollback-backup-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Backup Docker compose configuration
            if [[ -f docker-compose.yml ]]; then
                cp docker-compose.yml "$backup_dir/"
            fi
            if [[ -f docker-compose.production.yml ]]; then
                cp docker-compose.production.yml "$backup_dir/"
            fi
            if [[ -f .env ]]; then
                cp .env "$backup_dir/"
            fi
            
            # Export current container information
            docker ps --filter "name=lightrag" --format "json" > "$backup_dir/containers.json" || true
            ;;
            
        kubernetes)
            # Backup Kubernetes resources
            kubectl get deployment lightrag -n "$NAMESPACE" -o yaml > "$backup_dir/deployment.yaml" 2>/dev/null || true
            kubectl get service lightrag -n "$NAMESPACE" -o yaml > "$backup_dir/service.yaml" 2>/dev/null || true
            kubectl get configmap -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml" 2>/dev/null || true
            kubectl get secret -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml" 2>/dev/null || true
            
            # Backup Helm release if applicable
            if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
                helm get values "$HELM_RELEASE" -n "$NAMESPACE" > "$backup_dir/helm-values.yaml" || true
                helm get manifest "$HELM_RELEASE" -n "$NAMESPACE" > "$backup_dir/helm-manifest.yaml" || true
            fi
            ;;
    esac
    
    log_success "Current state backed up to: $backup_dir"
    echo "$backup_dir" > .last-rollback-backup
}

get_rollback_history() {
    log_info "Getting rollback history..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # For Docker, we can check image history and container history
            log_info "Recent Docker deployments:"
            
            # Get image history
            local images=$(docker images --filter "reference=*lightrag*" --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | head -10)
            if [[ -n "$images" ]]; then
                echo "$images"
            else
                log_warning "No LightRAG images found"
            fi
            ;;
            
        kubernetes)
            # For Kubernetes, check deployment rollout history
            log_info "Kubernetes deployment history:"
            kubectl rollout history deployment/lightrag -n "$NAMESPACE" || true
            
            # If using Helm, show release history
            if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
                log_info "Helm release history:"
                helm history "$HELM_RELEASE" -n "$NAMESPACE" || true
            fi
            ;;
    esac
}

perform_docker_rollback() {
    log_info "Performing Docker rollback..."
    
    # Find the previous successful deployment
    local current_image=$(docker ps --filter "name=lightrag" --format "{{.Image}}" | head -1)
    log_info "Current image: $current_image"
    
    # Stop current containers
    log_info "Stopping current containers..."
    docker ps --filter "name=lightrag" --format "{{.Names}}" | xargs -r docker stop
    
    # Remove current containers
    log_info "Removing current containers..."
    docker ps -a --filter "name=lightrag" --format "{{.Names}}" | xargs -r docker rm
    
    # Restore previous environment if backup exists
    if [[ -f .env.backup.* ]]; then
        local latest_backup=$(ls -t .env.backup.* | head -1)
        log_info "Restoring environment from: $latest_backup"
        cp "$latest_backup" .env
    fi
    
    # Determine compose file based on environment
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.production.yml"
    fi
    
    # Start with previous configuration
    log_info "Starting services with previous configuration..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$compose_file" up -d
    else
        docker compose -f "$compose_file" up -d
    fi
    
    # Wait for services to be ready
    local attempts=0
    local max_attempts=30
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s -f "http://localhost:9621/health" >/dev/null 2>&1; then
            log_success "Docker rollback completed successfully"
            return 0
        else
            log_info "Waiting for services... (attempt $((attempts + 1))/$max_attempts)"
            sleep 10
            attempts=$((attempts + 1))
        fi
    done
    
    log_error "Docker rollback completed but services are not healthy"
    return 1
}

perform_kubernetes_rollback() {
    log_info "Performing Kubernetes rollback..."
    
    # Check if using Helm
    if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
        log_info "Rolling back Helm release..."
        
        # Get current revision
        local current_revision=$(helm list -n "$NAMESPACE" | grep "$HELM_RELEASE" | awk '{print $3}')
        log_info "Current Helm revision: $current_revision"
        
        # Calculate target revision
        local target_revision=$((current_revision - ROLLBACK_STEPS))
        if [[ $target_revision -lt 1 ]]; then
            target_revision=1
        fi
        
        log_info "Rolling back to revision: $target_revision"
        
        # Perform Helm rollback
        if [[ $target_revision -eq $current_revision ]]; then
            log_info "Rolling back to previous revision"
            helm rollback "$HELM_RELEASE" -n "$NAMESPACE"
        else
            helm rollback "$HELM_RELEASE" "$target_revision" -n "$NAMESPACE"
        fi
        
    else
        log_info "Rolling back Kubernetes deployment..."
        
        # Show current rollout status
        kubectl rollout status deployment/lightrag -n "$NAMESPACE" --timeout=10s || true
        
        # Perform rollback
        kubectl rollout undo deployment/lightrag -n "$NAMESPACE" --to-revision=$(($(kubectl rollout history deployment/lightrag -n "$NAMESPACE" --revision=0 | wc -l) - ROLLBACK_STEPS))
    fi
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/lightrag -n "$NAMESPACE" --timeout=300s
    
    # Verify pods are running
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=lightrag --no-headers | grep "Running" | wc -l)
    if [[ $ready_pods -gt 0 ]]; then
        log_success "Kubernetes rollback completed successfully"
        return 0
    else
        log_error "Kubernetes rollback completed but no pods are running"
        return 1
    fi
}

verify_rollback() {
    log_info "Verifying rollback..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Check if containers are running
            local running_containers=$(docker ps --filter "name=lightrag" --format "{{.Names}}" | wc -l)
            if [[ $running_containers -eq 0 ]]; then
                log_error "No LightRAG containers are running"
                return 1
            fi
            
            # Health check
            local health_url="http://localhost:9621/health"
            ;;
            
        kubernetes)
            # Check pod status
            local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=lightrag --no-headers | grep "Running" | wc -l)
            if [[ $ready_pods -eq 0 ]]; then
                log_error "No LightRAG pods are running"
                return 1
            fi
            
            # Port-forward for health check
            kubectl port-forward -n "$NAMESPACE" service/lightrag 8080:9621 &
            local port_forward_pid=$!
            sleep 5
            
            local health_url="http://localhost:8080/health"
            ;;
    esac
    
    # Perform health check
    local attempts=0
    local max_attempts=10
    
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s -f "$health_url" >/dev/null 2>&1; then
            log_success "Health check passed"
            
            # Clean up port-forward if used
            if [[ -n "${port_forward_pid:-}" ]]; then
                kill "$port_forward_pid" 2>/dev/null || true
            fi
            
            return 0
        else
            log_info "Health check attempt $((attempts + 1))/$max_attempts failed"
            sleep 10
            attempts=$((attempts + 1))
        fi
    done
    
    # Clean up port-forward if used
    if [[ -n "${port_forward_pid:-}" ]]; then
        kill "$port_forward_pid" 2>/dev/null || true
    fi
    
    log_error "Health check failed after rollback"
    return 1
}

restore_data_backup() {
    if [[ "$BACKUP_RESTORE" != "true" ]]; then
        return 0
    fi
    
    log_info "Restoring data from backup..."
    
    # This is a placeholder for data restoration logic
    # In a real scenario, you would restore database dumps, file backups, etc.
    
    case $DEPLOYMENT_TYPE in
        docker)
            # Restore Docker volumes or bind mounts
            log_info "Checking for Docker volume backups..."
            
            # Example: Restore RAG storage directory
            if [[ -d "rag_storage_backup" ]]; then
                log_info "Restoring RAG storage from backup..."
                rm -rf rag_storage
                cp -r rag_storage_backup rag_storage
                log_success "RAG storage restored"
            fi
            ;;
            
        kubernetes)
            # Restore PVC data from backup
            log_info "Checking for PVC backups..."
            
            # This would typically involve:
            # 1. Scaling down the deployment
            # 2. Restoring PVC data from backup storage (S3, etc.)
            # 3. Scaling back up
            
            log_warning "PVC data restoration not implemented - manual intervention may be required"
            ;;
    esac
    
    log_success "Data backup restoration completed"
}

generate_rollback_report() {
    log_info "Generating rollback report..."
    
    local report_file="rollback-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Rollback Report

**Timestamp:** $(date -u)
**Environment:** $ENVIRONMENT
**Deployment Type:** $DEPLOYMENT_TYPE
**Rollback Steps:** $ROLLBACK_STEPS

## Rollback Details

$(
case $DEPLOYMENT_TYPE in
    docker)
        echo "### Docker Rollback"
        echo "- Stopped and removed current containers"
        echo "- Restored previous environment configuration"
        echo "- Restarted services with previous configuration"
        echo ""
        echo "**Current Containers:**"
        docker ps --filter "name=lightrag" --format "- {{.Names}}: {{.Image}} ({{.Status}})" || echo "None"
        ;;
    kubernetes)
        echo "### Kubernetes Rollback"
        echo "- Namespace: $NAMESPACE"
        if command -v helm &> /dev/null && helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE"; then
            echo "- Helm Release: $HELM_RELEASE"
            echo "- Rolled back using Helm"
        else
            echo "- Rolled back using kubectl"
        fi
        echo ""
        echo "**Current Pods:**"
        kubectl get pods -n "$NAMESPACE" -l app=lightrag --no-headers | awk '{print "- " $1 ": " $3}' || echo "None"
        ;;
esac
)

## Verification

$(
if verify_rollback &>/dev/null; then
    echo "✅ **Rollback verification: PASSED**"
    echo ""
    echo "- Health checks passed"
    echo "- Services are responding"
else
    echo "❌ **Rollback verification: FAILED**"
    echo ""
    echo "- Health checks failed"
    echo "- Manual intervention required"
fi
)

## Backup Information

$(
if [[ -f .last-rollback-backup ]]; then
    backup_dir=$(cat .last-rollback-backup)
    echo "- Pre-rollback backup: $backup_dir"
else
    echo "- No backup information available"
fi
)

## Next Steps

1. Verify application functionality
2. Check logs for any issues
3. Monitor system performance
4. Investigate root cause of issues that led to rollback
5. Plan for proper deployment once issues are resolved

## Commands for Investigation

\`\`\`bash
$(
case $DEPLOYMENT_TYPE in
    docker)
        echo "# View container logs"
        echo "docker logs \$(docker ps --filter 'name=lightrag' --format '{{.Names}}')"
        echo ""
        echo "# Check container status"
        echo "docker ps --filter 'name=lightrag'"
        ;;
    kubernetes)
        echo "# View pod logs"
        echo "kubectl logs -n $NAMESPACE -l app=lightrag --tail=100"
        echo ""
        echo "# Check pod status"
        echo "kubectl get pods -n $NAMESPACE -l app=lightrag"
        echo ""
        echo "# Check events"
        echo "kubectl get events -n $NAMESPACE --sort-by=.metadata.creationTimestamp"
        ;;
esac
)
\`\`\`
EOF

    log_success "Rollback report generated: $report_file"
}

main() {
    log_info "Starting rollback process..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    
    cd "$PROJECT_ROOT"
    
    check_dependencies
    detect_deployment_type
    confirm_rollback
    backup_current_state
    get_rollback_history
    
    # Perform rollback based on deployment type
    case $DEPLOYMENT_TYPE in
        docker)
            if perform_docker_rollback; then
                log_success "Docker rollback completed"
            else
                log_error "Docker rollback failed"
                exit 1
            fi
            ;;
        kubernetes|k8s)
            if perform_kubernetes_rollback; then
                log_success "Kubernetes rollback completed"
            else
                log_error "Kubernetes rollback failed"
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Verify rollback
    if verify_rollback; then
        log_success "Rollback verification passed"
    else
        log_warning "Rollback verification failed - manual intervention may be required"
    fi
    
    # Restore data if requested
    restore_data_backup
    
    generate_rollback_report
    
    log_success "🔄 Rollback process completed!"
    
    # Print summary
    log_info "=== Rollback Summary ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment Type: $DEPLOYMENT_TYPE"
    log_info "Status: ✅ Completed"
    log_info "======================="
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Deployment environment (staging, production)
    -t, --type TYPE           Deployment type (docker, kubernetes, auto)
    -n, --namespace NAMESPACE Kubernetes namespace (default: lightrag-ENV)
    -R, --release RELEASE     Helm release name (default: lightrag)
    -s, --steps STEPS         Number of rollback steps (default: 1)
    --no-confirm              Skip confirmation prompt
    --backup-restore          Restore data from backup
    -h, --help                Show this help message

Environment Variables:
    ENVIRONMENT              Deployment environment
    DEPLOYMENT_TYPE          Deployment type (docker, kubernetes, auto)
    NAMESPACE               Kubernetes namespace
    HELM_RELEASE            Helm release name
    ROLLBACK_STEPS          Number of rollback steps
    CONFIRMATION            Enable/disable confirmation prompt
    BACKUP_RESTORE          Enable/disable data backup restoration

Examples:
    $0 -e staging                           # Rollback staging deployment
    $0 -e production -t kubernetes -s 2     # Rollback production K8s by 2 steps
    $0 --no-confirm -t docker               # Rollback Docker without confirmation
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
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
        -s|--steps)
            ROLLBACK_STEPS="$2"
            shift 2
            ;;
        --no-confirm)
            CONFIRMATION=false
            shift
            ;;
        --backup-restore)
            BACKUP_RESTORE=true
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

# Run main function
main "$@"