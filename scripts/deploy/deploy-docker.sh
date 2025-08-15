#!/bin/bash
set -euo pipefail

# Docker deployment script
# This script automates Docker deployments for LightRAG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration with defaults
ENVIRONMENT=${ENVIRONMENT:-staging}
IMAGE_TAG=${IMAGE_TAG:-latest}
REGISTRY=${REGISTRY:-ghcr.io}
IMAGE_NAME=${IMAGE_NAME:-hkuds/lightrag}
COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
DEPLOYMENT_TIMEOUT=${DEPLOYMENT_TIMEOUT:-300}
HEALTH_CHECK_RETRIES=${HEALTH_CHECK_RETRIES:-30}
HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-10}

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
    log_info "Checking deployment dependencies..."

    local missing_deps=()

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        missing_deps+=("docker-compose")
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

    log_success "All dependencies found"
}

validate_environment() {
    log_info "Validating deployment environment: $ENVIRONMENT"

    case $ENVIRONMENT in
        development|dev)
            COMPOSE_FILE="docker-compose.yml"
            ;;
        staging|stage)
            COMPOSE_FILE="docker-compose.yml"
            ;;
        production|prod)
            COMPOSE_FILE="docker-compose.production.yml"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_info "Valid environments: development, staging, production"
            exit 1
            ;;
    esac

    log_success "Environment validated: $ENVIRONMENT"
}

prepare_environment_config() {
    log_info "Preparing environment configuration..."

    cd "$PROJECT_ROOT"

    # Backup existing .env if it exists
    if [[ -f .env ]]; then
        cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)"
        log_info "Backed up existing .env file"
    fi

    # Choose appropriate environment file
    local env_source=""
    case $ENVIRONMENT in
        production|prod)
            if [[ -f production.env ]]; then
                env_source="production.env"
            else
                log_error "production.env file not found"
                exit 1
            fi
            ;;
        staging|stage)
            if [[ -f staging.env ]]; then
                env_source="staging.env"
            elif [[ -f env.example ]]; then
                env_source="env.example"
            else
                log_error "No staging environment file found"
                exit 1
            fi
            ;;
        development|dev)
            if [[ -f env.example ]]; then
                env_source="env.example"
            else
                log_error "env.example file not found"
                exit 1
            fi
            ;;
    esac

    # Copy environment file
    cp "$env_source" .env

    # Update environment-specific variables
    case $ENVIRONMENT in
        production|prod)
            sed -i "s/NODE_ENV=.*/NODE_ENV=production/" .env
            sed -i "s/DEBUG=.*/DEBUG=false/" .env
            sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=INFO/" .env
            ;;
        staging|stage)
            sed -i "s/NODE_ENV=.*/NODE_ENV=staging/" .env
            sed -i "s/DEBUG=.*/DEBUG=false/" .env
            sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/" .env
            ;;
        development|dev)
            sed -i "s/NODE_ENV=.*/NODE_ENV=development/" .env
            sed -i "s/DEBUG=.*/DEBUG=true/" .env
            sed -i "s/LOG_LEVEL=.*/LOG_LEVEL=DEBUG/" .env
            ;;
    esac

    log_success "Environment configuration prepared from $env_source"
}

pull_docker_images() {
    log_info "Pulling Docker images..."

    local full_image_name="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    # Pull main application image
    if docker pull "$full_image_name"; then
        log_success "Pulled application image: $full_image_name"
    else
        log_error "Failed to pull application image: $full_image_name"
        exit 1
    fi

    # Pull dependency images specified in compose file
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_info "Pulling dependency images from $COMPOSE_FILE..."

        # Use docker-compose to pull all images
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" pull
        else
            docker compose -f "$COMPOSE_FILE" pull
        fi

        log_success "Dependency images pulled"
    fi
}

prepare_docker_compose() {
    log_info "Preparing Docker Compose configuration..."

    # Create deployment-specific compose file
    local deploy_compose="docker-compose.deploy.yml"

    # Copy base compose file
    cp "$COMPOSE_FILE" "$deploy_compose"

    # Update image tag in compose file
    local full_image_name="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

    # Replace image name with specific tag
    if command -v sed &> /dev/null; then
        # Update the lightrag service image
        sed -i "s|image: lightrag:.*|image: $full_image_name|g" "$deploy_compose"
        sed -i "s|build:.*|image: $full_image_name|g" "$deploy_compose"

        # Remove build contexts to use pulled images
        sed -i '/build:/,/dockerfile:/d' "$deploy_compose"
    fi

    log_success "Docker Compose configuration prepared: $deploy_compose"
    export DEPLOY_COMPOSE_FILE="$deploy_compose"
}

stop_existing_services() {
    log_info "Stopping existing services..."

    # Stop services using the deployment compose file
    if [[ -f "${DEPLOY_COMPOSE_FILE:-$COMPOSE_FILE}" ]]; then
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "${DEPLOY_COMPOSE_FILE:-$COMPOSE_FILE}" down || true
        else
            docker compose -f "${DEPLOY_COMPOSE_FILE:-$COMPOSE_FILE}" down || true
        fi
    fi

    # Stop any running containers by name pattern
    local containers=$(docker ps --filter "name=lightrag" --format "{{.Names}}" || true)
    if [[ -n "$containers" ]]; then
        log_info "Stopping existing LightRAG containers: $containers"
        echo "$containers" | xargs -r docker stop
    fi

    log_success "Existing services stopped"
}

start_services() {
    log_info "Starting services..."

    local compose_file="${DEPLOY_COMPOSE_FILE:-$COMPOSE_FILE}"

    # Start services
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$compose_file" up -d
    else
        docker compose -f "$compose_file" up -d
    fi

    log_success "Services started"
}

wait_for_services() {
    log_info "Waiting for services to be healthy..."

    # Determine service URLs based on environment
    local base_url="http://localhost:9621"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # In production, services might be behind a load balancer
        base_url="${PRODUCTION_URL:-http://localhost:9621}"
    fi

    # Wait for main application
    local health_url="$base_url/health"
    local attempts=0
    local max_attempts=$HEALTH_CHECK_RETRIES

    log_info "Checking health endpoint: $health_url"

    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s -f "$health_url" >/dev/null 2>&1; then
            log_success "Application is healthy"
            break
        else
            log_info "Waiting for application... (attempt $((attempts + 1))/$max_attempts)"
            sleep "$HEALTH_CHECK_INTERVAL"
            attempts=$((attempts + 1))
        fi
    done

    if [[ $attempts -eq $max_attempts ]]; then
        log_error "Application failed to become healthy within timeout"
        return 1
    fi

    # Test detailed health endpoint
    local detailed_health_url="$base_url/api/health"
    if curl -s -f "$detailed_health_url" >/dev/null 2>&1; then
        log_success "Detailed health check passed"
    else
        log_warning "Detailed health check failed (non-critical)"
    fi

    return 0
}

run_post_deployment_tests() {
    log_info "Running post-deployment tests..."

    local base_url="http://localhost:9621"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        base_url="${PRODUCTION_URL:-http://localhost:9621}"
    fi

    # Test health endpoints
    log_info "Testing health endpoints..."

    if ! curl -s -f "$base_url/health" >/dev/null; then
        log_error "Health endpoint test failed"
        return 1
    fi

    if ! curl -s -f "$base_url/api/health" >/dev/null; then
        log_warning "Detailed health endpoint test failed"
    fi

    # Test basic API functionality (if not production)
    if [[ "$ENVIRONMENT" != "production" ]]; then
        log_info "Testing basic API functionality..."

        # Test a simple query (assuming no authentication required in staging)
        local query_response=$(curl -s -X POST "$base_url/query" \
            -H "Content-Type: application/json" \
            -d '{"query": "test", "mode": "naive"}' || echo "failed")

        if [[ "$query_response" != "failed" ]]; then
            log_success "Basic API test passed"
        else
            log_warning "Basic API test failed (may be due to missing LLM keys)"
        fi
    fi

    log_success "Post-deployment tests completed"
}

collect_deployment_info() {
    log_info "Collecting deployment information..."

    # Create deployment info file
    cat > deployment-info.json << EOF
{
  "deployment": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$ENVIRONMENT",
    "image_tag": "$IMAGE_TAG",
    "image_name": "$REGISTRY/$IMAGE_NAME:$IMAGE_TAG",
    "compose_file": "${DEPLOY_COMPOSE_FILE:-$COMPOSE_FILE}",
    "deployed_by": "${USER:-unknown}",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')"
  },
  "services": {}
}
EOF

    # Collect container information
    if command -v jq &> /dev/null; then
        local containers=$(docker ps --filter "label=com.docker.compose.project" --format "{{.Names}}")

        for container in $containers; do
            local container_info=$(docker inspect "$container" --format '{{json .}}' | jq '{
                "id": .Id[0:12],
                "image": .Config.Image,
                "status": .State.Status,
                "started_at": .State.StartedAt,
                "ports": [.NetworkSettings.Ports | to_entries[] | select(.value != null) | .key]
            }')

            # Add to deployment info
            jq --arg name "$container" --argjson info "$container_info" \
                '.services[$name] = $info' deployment-info.json > deployment-info.tmp && \
                mv deployment-info.tmp deployment-info.json
        done
    fi

    log_success "Deployment information collected: deployment-info.json"
}

cleanup_deployment_files() {
    log_info "Cleaning up temporary deployment files..."

    # Remove temporary compose file
    if [[ -n "${DEPLOY_COMPOSE_FILE:-}" ]] && [[ -f "$DEPLOY_COMPOSE_FILE" ]]; then
        rm -f "$DEPLOY_COMPOSE_FILE"
    fi

    log_success "Cleanup completed"
}

rollback() {
    log_error "Deployment failed - initiating rollback..."

    # Stop current deployment
    stop_existing_services

    # Restore previous environment
    if [[ -f .env.backup.* ]]; then
        local latest_backup=$(ls -t .env.backup.* | head -1)
        cp "$latest_backup" .env
        log_info "Restored environment from $latest_backup"
    fi

    # Try to restart with previous configuration
    if [[ -f "$COMPOSE_FILE" ]]; then
        log_info "Attempting to restart with previous configuration..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -f "$COMPOSE_FILE" up -d || true
        else
            docker compose -f "$COMPOSE_FILE" up -d || true
        fi
    fi

    log_error "Rollback completed - manual intervention may be required"
}

main() {
    log_info "Starting Docker deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Image: $REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
    log_info "Compose file: $COMPOSE_FILE"

    # Set up error handling
    trap 'rollback' ERR

    cd "$PROJECT_ROOT"

    check_dependencies
    validate_environment
    prepare_environment_config
    pull_docker_images
    prepare_docker_compose
    stop_existing_services
    start_services

    if wait_for_services; then
        run_post_deployment_tests
        collect_deployment_info
        log_success "🚀 Deployment completed successfully!"
    else
        log_error "Deployment failed - services did not become healthy"
        exit 1
    fi

    cleanup_deployment_files

    # Print deployment summary
    log_info "=== Deployment Summary ==="
    log_info "Environment: $ENVIRONMENT"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Services: $(docker ps --filter 'label=com.docker.compose.project' --format '{{.Names}}' | tr '\n' ' ')"
    log_info "Health Check: ✅"
    log_info "=========================="
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV     Deployment environment (development, staging, production)
    -t, --tag TAG             Docker image tag (default: latest)
    -r, --registry REGISTRY   Docker registry (default: ghcr.io)
    -i, --image IMAGE         Image name (default: hkuds/lightrag)
    -f, --file FILE           Docker Compose file (auto-selected based on environment)
    -T, --timeout SECONDS     Deployment timeout (default: 300)
    -h, --help                Show this help message

Environment Variables:
    ENVIRONMENT              Deployment environment
    IMAGE_TAG               Docker image tag
    REGISTRY                Docker registry
    IMAGE_NAME              Docker image name
    COMPOSE_FILE            Docker Compose file
    DEPLOYMENT_TIMEOUT      Deployment timeout in seconds
    HEALTH_CHECK_RETRIES    Number of health check retries
    HEALTH_CHECK_INTERVAL   Health check interval in seconds

Examples:
    $0 -e staging -t v1.2.3                    # Deploy specific version to staging
    $0 -e production -t latest                   # Deploy latest to production
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
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -T|--timeout)
            DEPLOYMENT_TIMEOUT="$2"
            shift 2
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
