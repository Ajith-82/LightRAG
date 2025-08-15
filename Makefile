# Makefile for LightRAG
# This Makefile provides automation for build, test, and deployment tasks

.PHONY: help install install-dev install-api install-test clean clean-all
.PHONY: test test-unit test-integration test-coverage test-performance
.PHONY: lint format security check quality
.PHONY: build build-package build-docker build-frontend
.PHONY: deploy deploy-docker deploy-k8s rollback health-check
.PHONY: docs docs-serve release

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip3
PROJECT_NAME := lightrag-hku
DOCKER_IMAGE := lightrag
DOCKER_TAG := latest
DOCKER_REGISTRY := ghcr.io/hkuds/lightrag
ENVIRONMENT := staging
NAMESPACE := lightrag-$(ENVIRONMENT)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Helper functions
define log_info
	@echo -e "$(BLUE)[INFO]$(NC) $(1)"
endef

define log_success
	@echo -e "$(GREEN)[SUCCESS]$(NC) $(1)"
endef

define log_warning
	@echo -e "$(YELLOW)[WARNING]$(NC) $(1)"
endef

define log_error
	@echo -e "$(RED)[ERROR]$(NC) $(1)"
endef

help: ## Show this help message
	@echo "LightRAG Build Automation"
	@echo "========================"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Environment variables:"
	@echo "  ENVIRONMENT  Deployment environment (default: staging)"
	@echo "  PYTHON       Python executable (default: python3)"
	@echo "  DOCKER_TAG   Docker image tag (default: latest)"

# =============================================================================
# INSTALLATION TARGETS
# =============================================================================

install: ## Install the package in production mode
	$(call log_info,Installing LightRAG...)
	$(PIP) install .
	$(call log_success,Installation completed)

install-dev: ## Install in development mode with all dependencies
	$(call log_info,Installing LightRAG in development mode...)
	$(PIP) install -e ".[test,api]"
	$(call log_success,Development installation completed)

install-api: ## Install with API dependencies
	$(call log_info,Installing LightRAG with API dependencies...)
	$(PIP) install -e ".[api]"
	$(call log_success,API installation completed)

install-test: ## Install test dependencies
	$(call log_info,Installing test dependencies...)
	$(PIP) install -e ".[test]"
	$(call log_success,Test dependencies installed)

# =============================================================================
# CLEANUP TARGETS
# =============================================================================

clean: ## Clean build artifacts and cache
	$(call log_info,Cleaning build artifacts...)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	$(call log_success,Cleanup completed)

clean-all: clean ## Clean everything including Docker images and volumes
	$(call log_info,Performing deep cleanup...)
	docker system prune -f || true
	docker volume prune -f || true
	rm -rf rag_storage/
	rm -rf test_*_storage/
	$(call log_success,Deep cleanup completed)

# =============================================================================
# TESTING TARGETS
# =============================================================================

test: ## Run all tests
	$(call log_info,Running all tests...)
	$(PYTHON) -m pytest tests/ -v --tb=short
	$(call log_success,All tests completed)

test-unit: ## Run unit tests only
	$(call log_info,Running unit tests...)
	$(PYTHON) -m pytest tests/ -m "unit and not slow" -v --tb=short
	$(call log_success,Unit tests completed)

test-integration: ## Run integration tests
	$(call log_info,Running integration tests...)
	./scripts/ci/integration-tests.sh
	$(call log_success,Integration tests completed)

test-coverage: ## Run tests with coverage report
	$(call log_info,Running tests with coverage...)
	$(PYTHON) -m pytest tests/ \
		--cov=lightrag \
		--cov=lightrag_mcp \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		-v
	$(call log_success,Coverage tests completed)

test-performance: ## Run performance tests
	$(call log_info,Running performance tests...)
	$(PYTHON) -m pytest tests/production/test_performance.py \
		--benchmark-only \
		--benchmark-json=benchmark-results.json \
		-v
	$(call log_success,Performance tests completed)

# =============================================================================
# CODE QUALITY TARGETS
# =============================================================================

lint: ## Run linting checks
	$(call log_info,Running linting checks...)
	./scripts/ci/lint-and-format.sh --check-only
	$(call log_success,Linting completed)

format: ## Format code automatically
	$(call log_info,Formatting code...)
	./scripts/ci/lint-and-format.sh --fix
	$(call log_success,Code formatting completed)

security: ## Run security scans
	$(call log_info,Running security scans...)
	./scripts/ci/security-audit.sh
	$(call log_success,Security scans completed)

check: lint test-unit ## Run basic quality checks (lint + unit tests)
	$(call log_success,Basic quality checks passed)

quality: lint test-coverage security ## Run comprehensive quality checks
	$(call log_success,Comprehensive quality checks completed)

# =============================================================================
# BUILD TARGETS
# =============================================================================

build: build-package build-docker ## Build everything (package + docker)

build-package: ## Build Python package
	$(call log_info,Building Python package...)
	$(PYTHON) -m build
	$(call log_success,Python package built)

build-docker: ## Build Docker image
	$(call log_info,Building Docker image...)
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build -t $(DOCKER_IMAGE):production-$(DOCKER_TAG) -f Dockerfile.production .
	$(call log_success,Docker images built)

build-frontend: ## Build frontend assets
	$(call log_info,Building frontend...)
	cd lightrag_webui && \
	(command -v bun >/dev/null 2>&1 && bun run build || npm run build-no-bun)
	$(call log_success,Frontend built)

# =============================================================================
# DEPLOYMENT TARGETS
# =============================================================================

deploy: deploy-docker ## Deploy to current environment

deploy-docker: ## Deploy using Docker
	$(call log_info,Deploying with Docker to $(ENVIRONMENT)...)
	./scripts/deploy/deploy-docker.sh -e $(ENVIRONMENT) -t $(DOCKER_TAG)
	$(call log_success,Docker deployment completed)

deploy-k8s: ## Deploy to Kubernetes
	$(call log_info,Deploying to Kubernetes...)
	./scripts/deploy/deploy-k8s.sh -e $(ENVIRONMENT) -t $(DOCKER_TAG) -n $(NAMESPACE)
	$(call log_success,Kubernetes deployment completed)

rollback: ## Rollback deployment
	$(call log_warning,Rolling back $(ENVIRONMENT) deployment...)
	./scripts/deploy/rollback.sh -e $(ENVIRONMENT)
	$(call log_success,Rollback completed)

health-check: ## Run health checks
	$(call log_info,Running health checks...)
	./scripts/deploy/health-check.sh -e $(ENVIRONMENT) --comprehensive
	$(call log_success,Health checks completed)

# =============================================================================
# DEVELOPMENT TARGETS
# =============================================================================

dev-setup: install-dev ## Set up development environment
	$(call log_info,Setting up development environment...)
	cp env.example .env || true
	$(call log_success,Development environment ready)

dev-server: ## Start development server
	$(call log_info,Starting development server...)
	$(PYTHON) -m lightrag.api.lightrag_server

dev-test: ## Run development tests (fast)
	$(call log_info,Running development tests...)
	$(PYTHON) -m pytest tests/ -x -v --tb=short -m "not slow"

dev-watch: ## Watch for changes and run tests
	$(call log_info,Watching for changes...)
	$(PYTHON) -m pytest-watch -- tests/ -x -v --tb=short -m "not slow"

# =============================================================================
# CI/CD TARGETS
# =============================================================================

ci-setup: ## Setup CI environment
	$(call log_info,Setting up CI environment...)
	$(PIP) install -e ".[test,api]"
	$(call log_success,CI environment ready)

ci-test: ## Run CI test suite
	$(call log_info,Running CI test suite...)
	./scripts/ci/check-coverage.sh
	./scripts/ci/lint-and-format.sh
	./scripts/ci/security-audit.sh
	$(call log_success,CI tests completed)

ci-build: ## Build for CI/CD
	$(call log_info,Building for CI/CD...)
	$(PYTHON) -m build
	docker build -t $(DOCKER_REGISTRY):$(DOCKER_TAG) -f Dockerfile.production .
	$(call log_success,CI build completed)

# =============================================================================
# DOCUMENTATION TARGETS
# =============================================================================

docs: ## Generate documentation
	$(call log_info,Generating documentation...)
	$(PYTHON) -m pydoc -w lightrag
	$(call log_success,Documentation generated)

docs-serve: ## Serve documentation locally
	$(call log_info,Serving documentation...)
	$(PYTHON) -m http.server 8000 -d docs/

# =============================================================================
# DATABASE TARGETS
# =============================================================================

db-setup: ## Set up development databases
	$(call log_info,Setting up development databases...)
	docker-compose up -d postgres redis
	sleep 10
	$(call log_success,Development databases ready)

db-clean: ## Clean database data
	$(call log_info,Cleaning database data...)
	docker-compose down -v
	rm -rf rag_storage/
	$(call log_success,Database data cleaned)

db-backup: ## Backup database data
	$(call log_info,Backing up database data...)
	./backup/backup-script.sh
	$(call log_success,Database backup completed)

# =============================================================================
# RELEASE TARGETS
# =============================================================================

release-check: ## Check if ready for release
	$(call log_info,Checking release readiness...)
	./scripts/ci/check-coverage.sh -t 80
	./scripts/ci/lint-and-format.sh
	./scripts/ci/security-audit.sh
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*
	$(call log_success,Release checks passed)

release-build: ## Build release artifacts
	$(call log_info,Building release artifacts...)
	rm -rf dist/
	$(PYTHON) -m build
	docker build -t $(DOCKER_REGISTRY):$(DOCKER_TAG) -f Dockerfile.production .
	$(call log_success,Release artifacts built)

release-publish: ## Publish to PyPI (requires authentication)
	$(call log_warning,Publishing to PyPI...)
	$(PYTHON) -m twine upload dist/*
	$(call log_success,Published to PyPI)

# =============================================================================
# MONITORING TARGETS
# =============================================================================

logs: ## View application logs
	@if docker ps --filter "name=lightrag" --format "{{.Names}}" | grep -q lightrag; then \
		echo "$(BLUE)[INFO]$(NC) Showing Docker logs..."; \
		docker logs -f $$(docker ps --filter "name=lightrag" --format "{{.Names}}" | head -1); \
	elif command -v kubectl >/dev/null 2>&1 && kubectl get pods -n $(NAMESPACE) -l app=lightrag >/dev/null 2>&1; then \
		echo "$(BLUE)[INFO]$(NC) Showing Kubernetes logs..."; \
		kubectl logs -f -n $(NAMESPACE) -l app=lightrag; \
	else \
		echo "$(RED)[ERROR]$(NC) No running deployment found"; \
	fi

status: ## Show deployment status
	@echo "$(BLUE)[INFO]$(NC) Deployment Status"
	@echo "===================="
	@if docker ps --filter "name=lightrag" --format "{{.Names}}" | grep -q lightrag; then \
		echo "$(GREEN)Docker Deployment:$(NC)"; \
		docker ps --filter "name=lightrag" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
	fi
	@if command -v kubectl >/dev/null 2>&1 && kubectl get namespace $(NAMESPACE) >/dev/null 2>&1; then \
		echo "$(GREEN)Kubernetes Deployment:$(NC)"; \
		kubectl get pods -n $(NAMESPACE) -l app=lightrag; \
	fi

# =============================================================================
# UTILITY TARGETS
# =============================================================================

version: ## Show version information
	@echo "LightRAG Version Information"
	@echo "==========================="
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Package: $$($(PYTHON) -c 'import lightrag; print(lightrag.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not available')"
	@echo "Kubectl: $$(kubectl version --client --short 2>/dev/null || echo 'Not available')"

env: ## Show environment information
	@echo "Environment Information"
	@echo "======================"
	@echo "ENVIRONMENT: $(ENVIRONMENT)"
	@echo "PYTHON: $(PYTHON)"
	@echo "DOCKER_TAG: $(DOCKER_TAG)"
	@echo "NAMESPACE: $(NAMESPACE)"
	@echo "PROJECT_NAME: $(PROJECT_NAME)"

# =============================================================================
# VALIDATION TARGETS
# =============================================================================

validate: ## Validate configuration and setup
	$(call log_info,Validating configuration...)
	@if [ ! -f pyproject.toml ]; then echo "$(RED)[ERROR]$(NC) pyproject.toml not found"; exit 1; fi
	@if [ ! -f .env ] && [ ! -f env.example ]; then echo "$(YELLOW)[WARNING]$(NC) No .env or env.example found"; fi
	@$(PYTHON) -c "import sys; print(f'Python version: {sys.version}')"
	@$(PIP) check || true
	$(call log_success,Configuration validated)

# =============================================================================
# SPECIAL TARGETS
# =============================================================================

all: clean install-dev test build ## Do everything (clean, install, test, build)

quick: test-unit lint ## Quick checks (unit tests + linting)

full: clean install-dev quality build-package ## Full validation (quality + build)

# Make targets that don't represent files
.PHONY: help install install-dev install-api install-test clean clean-all
.PHONY: test test-unit test-integration test-coverage test-performance
.PHONY: lint format security check quality
.PHONY: build build-package build-docker build-frontend
.PHONY: deploy deploy-docker deploy-k8s rollback health-check
.PHONY: dev-setup dev-server dev-test dev-watch
.PHONY: ci-setup ci-test ci-build
.PHONY: docs docs-serve
.PHONY: db-setup db-clean db-backup
.PHONY: release-check release-build release-publish
.PHONY: logs status version env validate
.PHONY: all quick full