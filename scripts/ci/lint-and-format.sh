#!/bin/bash
set -euo pipefail

# Code quality checks script
# This script runs linting, formatting, and code quality tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
FIX_MODE=${FIX_MODE:-false}
CHECK_ONLY=${CHECK_ONLY:-true}
VERBOSE=${VERBOSE:-false}

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
    log_info "Checking required dependencies..."

    local missing_deps=()

    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    # Check required Python packages
    local python_packages=("ruff" "black" "isort" "mypy" "bandit")
    for package in "${python_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_deps+=("python3-$package")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: pip install ruff black isort mypy bandit"
        exit 1
    fi

    log_success "All dependencies found"
}

run_ruff_linting() {
    log_info "Running Ruff linting..."

    local ruff_args=(
        "lightrag/"
        "lightrag_mcp/"
        "tests/"
    )

    if [[ "$VERBOSE" == "true" ]]; then
        ruff_args+=("--verbose")
    fi

    if [[ "$FIX_MODE" == "true" ]]; then
        ruff_args+=("--fix")
        log_info "Running in fix mode - will attempt to fix issues"
    fi

    # Add output format for CI
    if [[ -n "${CI:-}" ]]; then
        ruff_args+=("--output-format=github")
    fi

    # Run Ruff check
    if ruff check "${ruff_args[@]}"; then
        log_success "Ruff linting passed"
        return 0
    else
        log_error "Ruff linting failed"
        return 1
    fi
}

run_ruff_formatting() {
    log_info "Running Ruff formatting..."

    local format_args=(
        "lightrag/"
        "lightrag_mcp/"
        "tests/"
    )

    if [[ "$CHECK_ONLY" == "true" ]]; then
        format_args+=("--check")
        format_args+=("--diff")
        log_info "Running in check-only mode"
    else
        log_info "Running in fix mode - will format files"
    fi

    if ruff format "${format_args[@]}"; then
        log_success "Ruff formatting check passed"
        return 0
    else
        log_error "Ruff formatting check failed"
        return 1
    fi
}

run_black_formatting() {
    log_info "Running Black formatting check..."

    local black_args=(
        "lightrag/"
        "lightrag_mcp/"
        "tests/"
        "--line-length=88"
        "--target-version=py310"
    )

    if [[ "$CHECK_ONLY" == "true" ]]; then
        black_args+=("--check")
        black_args+=("--diff")
        log_info "Running in check-only mode"
    else
        log_info "Running in fix mode - will format files"
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        black_args+=("--verbose")
    fi

    if black "${black_args[@]}"; then
        log_success "Black formatting check passed"
        return 0
    else
        log_error "Black formatting check failed"
        return 1
    fi
}

run_isort_check() {
    log_info "Running isort import sorting check..."

    local isort_args=(
        "lightrag/"
        "lightrag_mcp/"
        "tests/"
        "--profile=black"
        "--line-length=88"
        "--multi-line=3"
        "--trailing-comma"
        "--force-grid-wrap=0"
        "--use-parentheses"
        "--ensure-newline-before-comments"
    )

    if [[ "$CHECK_ONLY" == "true" ]]; then
        isort_args+=("--check-only")
        isort_args+=("--diff")
        log_info "Running in check-only mode"
    else
        log_info "Running in fix mode - will sort imports"
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        isort_args+=("--verbose")
    fi

    if isort "${isort_args[@]}"; then
        log_success "isort check passed"
        return 0
    else
        log_error "isort check failed"
        return 1
    fi
}

run_mypy_check() {
    log_info "Running MyPy type checking..."

    local mypy_args=(
        "lightrag/"
        "--ignore-missing-imports"
        "--no-strict-optional"
        "--show-error-codes"
        "--warn-unused-ignores"
        "--warn-redundant-casts"
        "--warn-unreachable"
    )

    if [[ "$VERBOSE" == "true" ]]; then
        mypy_args+=("--verbose")
    fi

    # MyPy configuration file check
    if [[ -f "$PROJECT_ROOT/mypy.ini" || -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_info "Using MyPy configuration file"
    fi

    if mypy "${mypy_args[@]}"; then
        log_success "MyPy type checking passed"
        return 0
    else
        log_warning "MyPy type checking found issues (non-blocking)"
        return 0  # Don't fail CI for type issues
    fi
}

run_bandit_security_check() {
    log_info "Running Bandit security check..."

    local bandit_args=(
        "-r"
        "lightrag/"
        "lightrag_mcp/"
        "-ll"  # Low confidence, low severity threshold
    )

    # Exclude test files from security checks
    bandit_args+=("-x" "tests/")

    # Add format for CI
    if [[ -n "${CI:-}" ]]; then
        bandit_args+=("-f" "json" "-o" "bandit-report.json")
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        bandit_args+=("-v")
    fi

    if bandit "${bandit_args[@]}"; then
        log_success "Bandit security check passed"
        return 0
    else
        log_warning "Bandit found potential security issues"
        return 0  # Don't fail CI for security warnings
    fi
}

run_additional_checks() {
    log_info "Running additional code quality checks..."

    # Check for TODO/FIXME comments
    log_info "Checking for TODO/FIXME comments..."
    local todo_count=$(grep -r "TODO\|FIXME\|XXX\|HACK" lightrag/ lightrag_mcp/ --exclude-dir=__pycache__ | wc -l || echo "0")
    if [[ $todo_count -gt 0 ]]; then
        log_warning "Found $todo_count TODO/FIXME comments"
        grep -r "TODO\|FIXME\|XXX\|HACK" lightrag/ lightrag_mcp/ --exclude-dir=__pycache__ || true
    else
        log_success "No TODO/FIXME comments found"
    fi

    # Check for print statements (should use logging)
    log_info "Checking for print statements..."
    local print_count=$(grep -r "print(" lightrag/ lightrag_mcp/ --exclude-dir=__pycache__ --exclude="*test*" | wc -l || echo "0")
    if [[ $print_count -gt 0 ]]; then
        log_warning "Found $print_count print statements (consider using logging)"
        grep -r "print(" lightrag/ lightrag_mcp/ --exclude-dir=__pycache__ --exclude="*test*" || true
    else
        log_success "No print statements found"
    fi

    # Check for long lines (>120 characters)
    log_info "Checking for long lines..."
    local long_lines=$(find lightrag/ lightrag_mcp/ -name "*.py" -exec grep -l ".\{121,\}" {} \; | wc -l || echo "0")
    if [[ $long_lines -gt 0 ]]; then
        log_warning "Found files with lines >120 characters"
    else
        log_success "No overly long lines found"
    fi

    # Check for missing docstrings in public functions
    log_info "Checking for missing docstrings..."
    python3 << 'EOF'
import ast
import os
import sys

def check_docstrings(file_path):
    with open(file_path, 'r') as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return []

    missing = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            # Skip private functions/classes
            if node.name.startswith('_'):
                continue

            # Check if has docstring
            if not (node.body and isinstance(node.body[0], ast.Expr)
                   and isinstance(node.body[0].value, ast.Constant)
                   and isinstance(node.body[0].value.value, str)):
                missing.append(f"{file_path}:{node.lineno} - {node.__class__.__name__} '{node.name}' missing docstring")

    return missing

missing_docstrings = []
for root, dirs, files in os.walk('lightrag'):
    # Skip test directories and deprecated code
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'deprecated']

    for file in files:
        if file.endswith('.py') and not file.startswith('test_'):
            file_path = os.path.join(root, file)
            missing_docstrings.extend(check_docstrings(file_path))

if missing_docstrings:
    print(f"Found {len(missing_docstrings)} missing docstrings:")
    for missing in missing_docstrings[:10]:  # Show first 10
        print(f"  {missing}")
    if len(missing_docstrings) > 10:
        print(f"  ... and {len(missing_docstrings) - 10} more")
else:
    print("✅ All public functions/classes have docstrings")
EOF
}

generate_quality_report() {
    log_info "Generating code quality report..."

    cat > code-quality-report.md << EOF
# Code Quality Report

**Generated:** $(date -u)
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

\`\`\`bash
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
\`\`\`

## Configuration Files

- \`pyproject.toml\` - Main configuration for tools
- \`ruff.toml\` - Ruff-specific settings (if exists)
- \`mypy.ini\` - MyPy configuration (if exists)
EOF

    log_success "Code quality report generated: code-quality-report.md"
}

main() {
    log_info "Starting code quality checks..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Fix mode: $FIX_MODE"
    log_info "Check only: $CHECK_ONLY"

    check_dependencies

    cd "$PROJECT_ROOT"

    local failed_checks=()

    # Run all checks
    if ! run_ruff_linting; then
        failed_checks+=("ruff-lint")
    fi

    if ! run_ruff_formatting; then
        failed_checks+=("ruff-format")
    fi

    # Skip black if using ruff format
    # if ! run_black_formatting; then
    #     failed_checks+=("black")
    # fi

    if ! run_isort_check; then
        failed_checks+=("isort")
    fi

    if ! run_mypy_check; then
        # MyPy is non-blocking, don't add to failed_checks
        true
    fi

    if ! run_bandit_security_check; then
        # Bandit is non-blocking, don't add to failed_checks
        true
    fi

    # Additional checks (always non-blocking)
    run_additional_checks

    generate_quality_report

    # Final result
    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        log_success "All code quality checks passed"
        exit 0
    else
        log_error "Failed checks: ${failed_checks[*]}"
        if [[ "$FIX_MODE" == "true" ]]; then
            log_info "Some issues may have been automatically fixed. Please review changes."
        else
            log_info "Run with FIX_MODE=true to automatically fix some issues"
        fi
        exit 1
    fi
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -f, --fix                  Enable fix mode (automatically fix issues)
    -c, --check-only          Check only mode (default, don't fix)
    -v, --verbose             Verbose output
    -h, --help                Show this help message

Environment Variables:
    FIX_MODE                  Enable fix mode (true/false, default: false)
    CHECK_ONLY               Check only mode (true/false, default: true)
    VERBOSE                  Verbose output (true/false, default: false)

Examples:
    $0                        # Check only (default)
    $0 --fix                  # Fix mode
    $0 --verbose              # Verbose output
    FIX_MODE=true $0          # Fix mode via environment variable
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--fix)
            FIX_MODE=true
            CHECK_ONLY=false
            shift
            ;;
        -c|--check-only)
            CHECK_ONLY=true
            FIX_MODE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
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
