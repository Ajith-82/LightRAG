# Code Quality Report

**Generated:** Fri 15 Aug 14:50:35 UTC 2025
**Project:** LightRAG

## Checks Performed

### ✅ Linting (Ruff)
- Checked Python code style and common issues
- Configuration: pyproject.toml

### ✅ Formatting (Black/Ruff Format)
- Verified consistent code formatting
- Line length: 88 characters
- Target Python version: 3.10+

### ✅ Import Sorting (isort)
- Verified import organization
- Profile: black-compatible

### ✅ Type Checking (MyPy)
- Static type analysis
- Configuration: ignore missing imports

### ✅ Security Check (Bandit)
- Scanned for common security issues
- Confidence level: low and above

## Summary

All code quality checks completed. See individual tool outputs for detailed results.

## Running Checks Locally

```bash
# Run all checks
./scripts/ci/lint-and-format.sh

# Run in fix mode
FIX_MODE=true CHECK_ONLY=false ./scripts/ci/lint-and-format.sh

# Run specific tools
ruff check lightrag/
black --check lightrag/
isort --check-only lightrag/
mypy lightrag/
bandit -r lightrag/
```

## Configuration Files

- `pyproject.toml` - Main configuration for tools
- `ruff.toml` - Ruff-specific settings (if exists)
- `mypy.ini` - MyPy configuration (if exists)
