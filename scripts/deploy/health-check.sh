#!/bin/bash
set -euo pipefail

# Post-deployment health check script
# This script validates deployment health and functionality

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration with defaults
ENVIRONMENT=${ENVIRONMENT:-staging}
BASE_URL=${BASE_URL:-http://localhost:9621}
NAMESPACE=${NAMESPACE:-lightrag-${ENVIRONMENT}}
TIMEOUT=${TIMEOUT:-60}
RETRIES=${RETRIES:-30}
INTERVAL=${INTERVAL:-10}
DEPLOYMENT_TYPE=${DEPLOYMENT_TYPE:-auto}
COMPREHENSIVE=${COMPREHENSIVE:-false}

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
    log_info "Checking health check dependencies..."

    local missing_deps=()

    # Check required tools
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    fi

    # Check jq for JSON processing
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found - JSON processing will be limited"
    fi

    # Check deployment-specific tools
    if [[ "$DEPLOYMENT_TYPE" =~ ^(kubernetes|k8s|auto)$ ]]; then
        if ! command -v kubectl &> /dev/null; then
            if [[ "$DEPLOYMENT_TYPE" =~ ^(kubernetes|k8s)$ ]]; then
                missing_deps+=("kubectl")
            fi
        fi
    fi

    if [[ "$DEPLOYMENT_TYPE" =~ ^(docker|auto)$ ]]; then
        if ! command -v docker &> /dev/null; then
            if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
                missing_deps+=("docker")
            fi
        fi
    fi

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
        if kubectl get namespace "$NAMESPACE" &> /dev/null 2>&1; then
            if kubectl get deployment lightrag -n "$NAMESPACE" &> /dev/null 2>&1; then
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

    # Default to direct HTTP if no deployment detected
    DEPLOYMENT_TYPE="direct"
    log_info "No specific deployment detected - using direct HTTP checks"
}

setup_port_forward() {
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        log_info "Setting up port-forward for Kubernetes health checks..."

        # Kill any existing port-forward
        pkill -f "kubectl.*port-forward.*lightrag" || true
        sleep 2

        # Start port-forward
        kubectl port-forward -n "$NAMESPACE" service/lightrag 8080:9621 &
        PORT_FORWARD_PID=$!

        # Update base URL
        BASE_URL="http://localhost:8080"

        # Wait for port-forward to be ready
        sleep 5

        log_success "Port-forward established (PID: $PORT_FORWARD_PID)"
    fi
}

cleanup_port_forward() {
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        log_info "Cleaning up port-forward..."
        kill "$PORT_FORWARD_PID" 2>/dev/null || true
        pkill -f "kubectl.*port-forward.*lightrag" || true
    fi
}

check_basic_health() {
    log_info "Checking basic health endpoint..."

    local health_url="$BASE_URL/health"
    local attempts=0

    while [[ $attempts -lt $RETRIES ]]; do
        log_info "Health check attempt $((attempts + 1))/$RETRIES: $health_url"

        if curl -s -f --max-time "$TIMEOUT" "$health_url" >/dev/null 2>&1; then
            log_success "Basic health check passed"
            return 0
        else
            log_warning "Health check failed, retrying in ${INTERVAL}s..."
            sleep "$INTERVAL"
            attempts=$((attempts + 1))
        fi
    done

    log_error "Basic health check failed after $RETRIES attempts"
    return 1
}

check_detailed_health() {
    log_info "Checking detailed health endpoint..."

    local detailed_health_url="$BASE_URL/api/health"

    local response=$(curl -s --max-time "$TIMEOUT" "$detailed_health_url" 2>/dev/null || echo "failed")

    if [[ "$response" == "failed" ]]; then
        log_warning "Detailed health endpoint not accessible"
        return 1
    fi

    # Parse health response if jq is available
    if command -v jq &> /dev/null; then
        local status=$(echo "$response" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
        local timestamp=$(echo "$response" | jq -r '.timestamp // "unknown"' 2>/dev/null || echo "unknown")

        log_info "Health Status: $status"
        log_info "Timestamp: $timestamp"

        # Check individual components
        local components=$(echo "$response" | jq -r '.components // {} | keys[]' 2>/dev/null || true)
        if [[ -n "$components" ]]; then
            log_info "Component status:"
            for component in $components; do
                local comp_status=$(echo "$response" | jq -r ".components.$component.status // \"unknown\"" 2>/dev/null || echo "unknown")
                log_info "  - $component: $comp_status"
            done
        fi

        if [[ "$status" == "healthy" ]]; then
            log_success "Detailed health check passed"
            return 0
        else
            log_warning "Detailed health check indicates issues: $status"
            return 1
        fi
    else
        log_success "Detailed health endpoint responded (jq not available for parsing)"
        return 0
    fi
}

check_api_functionality() {
    if [[ "$COMPREHENSIVE" != "true" ]]; then
        return 0
    fi

    log_info "Checking API functionality..."

    # Test basic query endpoint (without requiring LLM)
    local query_url="$BASE_URL/query"
    local test_payload='{"query": "test", "mode": "naive"}'

    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        --max-time "$TIMEOUT" \
        "$query_url" 2>/dev/null || echo "failed")

    if [[ "$response" == "failed" ]]; then
        log_warning "Query endpoint not accessible"
        return 1
    fi

    # Check if response contains expected structure
    if command -v jq &> /dev/null; then
        local has_response=$(echo "$response" | jq -e 'has("response")' 2>/dev/null || echo "false")
        local has_error=$(echo "$response" | jq -e 'has("error")' 2>/dev/null || echo "false")

        if [[ "$has_response" == "true" ]]; then
            log_success "Query endpoint functional"
            return 0
        elif [[ "$has_error" == "true" ]]; then
            local error_msg=$(echo "$response" | jq -r '.error // "unknown error"' 2>/dev/null)
            log_warning "Query endpoint returned error: $error_msg"
            return 1
        else
            log_warning "Query endpoint returned unexpected response"
            return 1
        fi
    else
        log_info "Query endpoint responded (jq not available for detailed parsing)"
        return 0
    fi
}

check_database_connectivity() {
    if [[ "$COMPREHENSIVE" != "true" ]]; then
        return 0
    fi

    log_info "Checking database connectivity..."

    case $DEPLOYMENT_TYPE in
        docker)
            # Check if database containers are running
            local postgres_container=$(docker ps --filter "name=postgres" --format "{{.Names}}" | head -1)
            local redis_container=$(docker ps --filter "name=redis" --format "{{.Names}}" | head -1)

            if [[ -n "$postgres_container" ]]; then
                if docker exec "$postgres_container" pg_isready >/dev/null 2>&1; then
                    log_success "PostgreSQL container is ready"
                else
                    log_warning "PostgreSQL container is not ready"
                fi
            else
                log_warning "PostgreSQL container not found"
            fi

            if [[ -n "$redis_container" ]]; then
                if docker exec "$redis_container" redis-cli ping >/dev/null 2>&1; then
                    log_success "Redis container is ready"
                else
                    log_warning "Redis container is not ready"
                fi
            else
                log_warning "Redis container not found"
            fi
            ;;

        kubernetes)
            # Check if database pods are running
            local postgres_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres --no-headers -o name | head -1)
            local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis --no-headers -o name | head -1)

            if [[ -n "$postgres_pod" ]]; then
                if kubectl exec -n "$NAMESPACE" "$postgres_pod" -- pg_isready >/dev/null 2>&1; then
                    log_success "PostgreSQL pod is ready"
                else
                    log_warning "PostgreSQL pod is not ready"
                fi
            else
                log_warning "PostgreSQL pod not found"
            fi

            if [[ -n "$redis_pod" ]]; then
                if kubectl exec -n "$NAMESPACE" "$redis_pod" -- redis-cli ping >/dev/null 2>&1; then
                    log_success "Redis pod is ready"
                else
                    log_warning "Redis pod is not ready"
                fi
            else
                log_warning "Redis pod not found"
            fi
            ;;
    esac
}

check_resource_usage() {
    if [[ "$COMPREHENSIVE" != "true" ]]; then
        return 0
    fi

    log_info "Checking resource usage..."

    case $DEPLOYMENT_TYPE in
        docker)
            # Check Docker container resource usage
            local lightrag_container=$(docker ps --filter "name=lightrag" --format "{{.Names}}" | head -1)

            if [[ -n "$lightrag_container" ]]; then
                log_info "Container resource usage:"
                docker stats "$lightrag_container" --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
            fi
            ;;

        kubernetes)
            # Check pod resource usage
            log_info "Pod resource usage:"
            kubectl top pods -n "$NAMESPACE" -l app=lightrag 2>/dev/null || log_warning "Metrics server not available"

            # Check resource limits and requests
            log_info "Resource limits and requests:"
            kubectl describe pods -n "$NAMESPACE" -l app=lightrag | grep -A 10 "Limits:\|Requests:" || true
            ;;
    esac
}

check_logs() {
    if [[ "$COMPREHENSIVE" != "true" ]]; then
        return 0
    fi

    log_info "Checking recent logs for errors..."

    case $DEPLOYMENT_TYPE in
        docker)
            local lightrag_container=$(docker ps --filter "name=lightrag" --format "{{.Names}}" | head -1)

            if [[ -n "$lightrag_container" ]]; then
                log_info "Recent container logs:"
                docker logs --tail=20 "$lightrag_container" 2>&1 | tail -10

                # Check for errors
                local error_count=$(docker logs --tail=100 "$lightrag_container" 2>&1 | grep -i "error\|exception\|failed" | wc -l)
                if [[ $error_count -gt 0 ]]; then
                    log_warning "Found $error_count error messages in recent logs"
                else
                    log_success "No errors found in recent logs"
                fi
            fi
            ;;

        kubernetes)
            log_info "Recent pod logs:"
            kubectl logs -n "$NAMESPACE" -l app=lightrag --tail=20 2>/dev/null | tail -10 || log_warning "Could not retrieve logs"

            # Check for errors
            local error_count=$(kubectl logs -n "$NAMESPACE" -l app=lightrag --tail=100 2>/dev/null | grep -i "error\|exception\|failed" | wc -l || echo "0")
            if [[ $error_count -gt 0 ]]; then
                log_warning "Found $error_count error messages in recent logs"
            else
                log_success "No errors found in recent logs"
            fi
            ;;
    esac
}

check_performance() {
    if [[ "$COMPREHENSIVE" != "true" ]]; then
        return 0
    fi

    log_info "Checking basic performance..."

    # Test response time
    local start_time=$(date +%s%N)
    if curl -s -f --max-time "$TIMEOUT" "$BASE_URL/health" >/dev/null 2>&1; then
        local end_time=$(date +%s%N)
        local response_time=$(((end_time - start_time) / 1000000)) # Convert to milliseconds

        log_info "Health endpoint response time: ${response_time}ms"

        if [[ $response_time -lt 1000 ]]; then
            log_success "Response time is good (<1s)"
        elif [[ $response_time -lt 5000 ]]; then
            log_warning "Response time is moderate (1-5s)"
        else
            log_warning "Response time is slow (>5s)"
        fi
    else
        log_warning "Could not measure response time"
    fi
}

generate_health_report() {
    log_info "Generating health check report..."

    local report_file="health-check-report-$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
  "health_check": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "deployment_type": "$DEPLOYMENT_TYPE",
    "base_url": "$BASE_URL",
    "comprehensive": $COMPREHENSIVE
  },
  "results": {
    "basic_health": $(check_basic_health &>/dev/null && echo "true" || echo "false"),
    "detailed_health": $(check_detailed_health &>/dev/null && echo "true" || echo "false"),
    "api_functionality": $(check_api_functionality &>/dev/null && echo "true" || echo "false"),
    "database_connectivity": "checked",
    "resource_usage": "checked",
    "logs": "checked",
    "performance": "checked"
  },
  "summary": {
    "overall_status": "$(
      if check_basic_health &>/dev/null && check_detailed_health &>/dev/null; then
        echo "healthy"
      else
        echo "unhealthy"
      fi
    )"
  }
}
EOF

    log_success "Health check report generated: $report_file"
}

main() {
    log_info "Starting post-deployment health checks..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Base URL: $BASE_URL"
    log_info "Comprehensive checks: $COMPREHENSIVE"

    # Set up cleanup
    trap cleanup_port_forward EXIT

    check_dependencies
    detect_deployment_type
    setup_port_forward

    local failed_checks=()

    # Core health checks
    if ! check_basic_health; then
        failed_checks+=("basic_health")
    fi

    if ! check_detailed_health; then
        failed_checks+=("detailed_health")
    fi

    # Additional checks
    if ! check_api_functionality; then
        failed_checks+=("api_functionality")
    fi

    # Comprehensive checks (non-blocking)
    check_database_connectivity || true
    check_resource_usage || true
    check_logs || true
    check_performance || true

    generate_health_report

    # Final result
    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        log_success "🏥 All health checks passed!"

        # Print summary
        log_info "=== Health Check Summary ==="
        log_info "Environment: $ENVIRONMENT"
        log_info "Deployment Type: $DEPLOYMENT_TYPE"
        log_info "Status: ✅ Healthy"
        log_info "============================"

        exit 0
    else
        log_error "❌ Health checks failed: ${failed_checks[*]}"

        # Print summary
        log_info "=== Health Check Summary ==="
        log_info "Environment: $ENVIRONMENT"
        log_info "Deployment Type: $DEPLOYMENT_TYPE"
        log_info "Status: ❌ Unhealthy"
        log_info "Failed: ${failed_checks[*]}"
        log_info "============================"

        exit 1
    fi
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Deployment environment (staging, production)
    -u, --url URL             Base URL for health checks (default: http://localhost:9621)
    -n, --namespace NAMESPACE Kubernetes namespace (default: lightrag-ENV)
    -t, --timeout SECONDS     Request timeout (default: 60)
    -r, --retries COUNT       Number of retries (default: 30)
    -i, --interval SECONDS    Retry interval (default: 10)
    -T, --type TYPE           Deployment type (docker, kubernetes, direct, auto)
    -c, --comprehensive       Run comprehensive health checks
    -h, --help                Show this help message

Environment Variables:
    ENVIRONMENT              Deployment environment
    BASE_URL                Base URL for health checks
    NAMESPACE               Kubernetes namespace
    TIMEOUT                 Request timeout in seconds
    RETRIES                 Number of retries
    INTERVAL                Retry interval in seconds
    DEPLOYMENT_TYPE         Deployment type
    COMPREHENSIVE           Enable comprehensive checks

Examples:
    $0 -e staging                              # Basic health check for staging
    $0 -e production -c                        # Comprehensive check for production
    $0 -u http://api.example.com -T direct     # Check external service
    $0 -n custom-namespace -T kubernetes       # Check custom K8s namespace
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -u|--url)
            BASE_URL="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -r|--retries)
            RETRIES="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -T|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -c|--comprehensive)
            COMPREHENSIVE=true
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
