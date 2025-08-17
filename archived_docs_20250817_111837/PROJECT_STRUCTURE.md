# LightRAG Project Structure

## Overview
Clean, organized repository structure for the LightRAG production-ready RAG system.

## Directory Structure

```
LightRAG/
├── .github/                    # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── ci.yml             # Main CI pipeline
│
├── assets/                     # Static assets for documentation
├── backup/                     # Backup scripts and configurations
│
├── docs/                       # Comprehensive documentation
│   ├── architecture/           # System architecture documentation
│   ├── ci-cd/                  # CI/CD documentation and reports
│   │   └── code-quality-report.md
│   ├── integration_guides/     # Integration and deployment guides
│   ├── production/             # Production deployment documentation
│   │   └── PRODUCTION_IMPLEMENTATION_GUIDELINES.md
│   ├── security/               # Security documentation
│   └── test_outputs/           # Test output examples
│
├── examples/                   # Example scripts and demos
│
├── k8s-deploy/                 # Kubernetes deployment configurations
│   ├── databases/              # Database deployment scripts
│   └── lightrag/               # LightRAG Kubernetes manifests
│
├── lightrag/                   # Main application code
│   ├── api/                    # FastAPI web server
│   │   ├── auth/              # Authentication system
│   │   ├── logging/           # Audit logging
│   │   ├── middleware/        # Security middleware
│   │   └── routers/           # API endpoints
│   ├── kg/                     # Storage backends
│   └── llm/                    # LLM integrations
│
├── lightrag_mcp/               # Model Context Protocol server
│   └── examples/               # MCP usage examples
│
├── lightrag_webui/             # React/TypeScript web UI
│
├── nginx/                      # Nginx configuration
├── postgres/                   # PostgreSQL configuration
├── redis/                      # Redis configuration
│
├── scripts/                    # Automation scripts
│   ├── ci/                     # CI/CD scripts
│   ├── deploy/                 # Deployment scripts
│   └── testing/                # Test automation
│
├── tests/                      # Comprehensive test suite
│   ├── core/                   # Core functionality tests
│   ├── integration/            # Integration tests
│   ├── production/             # Production readiness tests
│   ├── security/               # Security tests
│   └── storage/                # Storage backend tests
│
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore patterns
├── .gitlab-ci.yml              # GitLab CI configuration
├── .pre-commit-config.yaml     # Pre-commit hooks
├── .python-version             # Python version specification
├── CHANGELOG.md                # Version changelog
├── CLAUDE.md                   # Claude AI instructions
├── docker-compose.yml          # Development Docker setup
├── docker-compose.enhanced.yml # Enhanced PostgreSQL setup
├── docker-compose.production.yml # Production Docker setup
├── Dockerfile                  # Application container
├── Dockerfile.production       # Production container
├── LICENSE                     # MIT License
├── Makefile                    # Build automation
├── production.env              # Production environment template
├── pyproject.toml              # Python project configuration
├── README.md                   # Main documentation
├── setup.py                    # Python package setup
└── tox.ini                     # Test automation configuration
```

## Key Configuration Files

### Python Configuration
- `pyproject.toml` - Project dependencies and tool configuration
- `setup.py` - Package installation configuration
- `.python-version` - Python version for pyenv

### Docker Configuration
- `docker-compose.yml` - Development environment
- `docker-compose.enhanced.yml` - Enhanced PostgreSQL setup
- `docker-compose.production.yml` - Production deployment
- `Dockerfile` - Standard container build
- `Dockerfile.production` - Security-hardened production build

### CI/CD Configuration
- `.github/workflows/ci.yml` - GitHub Actions pipeline
- `.gitlab-ci.yml` - GitLab CI pipeline
- `.pre-commit-config.yaml` - Pre-commit hooks
- `tox.ini` - Test automation

### Environment Configuration
- `.env.example` - Example environment variables
- `production.env` - Production environment template
- `CLAUDE.md` - AI assistant instructions

## Test Coverage
- **424 total tests** across all test suites
- **Phase 1**: Infrastructure and security (24 tests)
- **Phase 2**: Core functionality (143 tests)
- **Phase 3**: Production hardening (78 tests)
- **Phase 4**: CI/CD automation (complete pipeline)

## Clean Repository Guidelines

### What's Ignored
- Python cache files (`__pycache__`, `*.pyc`)
- Test artifacts (`.pytest_cache`, `.coverage`, `htmlcov/`)
- IDE files (`.vscode/`, `.idea/`)
- Runtime data (`rag_storage/`, `logs/`)
- Environment files (`.env`, but not `.env.example`)
- Temporary files (`*.tmp`, `*.bak`, `*~`)

### Maintenance Commands
```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Clean test artifacts
rm -rf .pytest_cache/ .coverage htmlcov/ coverage.xml

# Clean all build artifacts
make clean-all

# Run pre-commit hooks
pre-commit run --all-files
```

## Development Workflow

1. **Setup Environment**
   ```bash
   pyenv install 3.12.10
   pyenv local 3.12.10
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[api,test]"
   ```

2. **Run Tests**
   ```bash
   make test
   make test-coverage
   ```

3. **Quality Checks**
   ```bash
   make quality
   make security
   ```

4. **Deploy**
   ```bash
   make deploy ENVIRONMENT=production
   ```

## Documentation Access
- Main README: `README.md`
- Documentation Index: `docs/DOCUMENTATION_INDEX.md`
- Production Guidelines: `docs/production/PRODUCTION_IMPLEMENTATION_GUIDELINES.md`
- API Documentation: Auto-generated at `/docs` endpoint
- Claude Instructions: `CLAUDE.md`

## Support
- GitHub Issues: Report bugs and feature requests
- Documentation: Comprehensive guides in `docs/`
- Examples: Working examples in `examples/`