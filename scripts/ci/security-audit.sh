#!/bin/bash
set -euo pipefail

# Security audit script
# This script runs comprehensive security scans and vulnerability assessments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
SCAN_TYPE=${SCAN_TYPE:-all}
OUTPUT_DIR=${OUTPUT_DIR:-security-reports}
FAIL_ON_HIGH=${FAIL_ON_HIGH:-true}
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
    log_info "Checking security scanning dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check required Python packages
    local python_packages=("safety" "bandit" "semgrep")
    for package in "${python_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            if [[ "$package" == "semgrep" ]]; then
                log_warning "Semgrep not found - will use pip-audit instead"
            else
                missing_deps+=("python3-$package")
            fi
        fi
    done
    
    # Check for external tools
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install Python packages with: pip install safety bandit pip-audit"
        exit 1
    fi
    
    log_success "Dependencies check completed"
}

setup_output_directory() {
    log_info "Setting up output directory: $OUTPUT_DIR"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Create subdirectories
    mkdir -p "$OUTPUT_DIR/dependency-scans"
    mkdir -p "$OUTPUT_DIR/code-scans"
    mkdir -p "$OUTPUT_DIR/secrets-scans"
    mkdir -p "$OUTPUT_DIR/container-scans"
    
    log_success "Output directory ready"
}

run_safety_scan() {
    log_info "Running Safety (PyUp) dependency vulnerability scan..."
    
    local safety_output="$OUTPUT_DIR/dependency-scans/safety-report"
    
    # JSON report
    if safety check --json --output "$safety_output.json" 2>/dev/null; then
        log_success "Safety scan completed - no vulnerabilities found"
        SAFETY_VULNERABILITIES=0
    else
        log_warning "Safety scan found vulnerabilities"
        SAFETY_VULNERABILITIES=$(jq '.vulnerabilities | length' "$safety_output.json" 2>/dev/null || echo "unknown")
    fi
    
    # Human-readable report
    safety check --output text > "$safety_output.txt" 2>&1 || true
    
    # Short report for console
    log_info "Safety scan summary:"
    safety check --short-report || true
    
    echo "$SAFETY_VULNERABILITIES" > "$OUTPUT_DIR/dependency-scans/safety-count.txt"
}

run_pip_audit_scan() {
    log_info "Running pip-audit vulnerability scan..."
    
    local audit_output="$OUTPUT_DIR/dependency-scans/pip-audit-report"
    
    # Install pip-audit if not available
    if ! command -v pip-audit &> /dev/null; then
        log_info "Installing pip-audit..."
        pip install pip-audit
    fi
    
    # JSON report
    if pip-audit --format=json --output="$audit_output.json" 2>/dev/null; then
        log_success "pip-audit scan completed - no vulnerabilities found"
        AUDIT_VULNERABILITIES=0
    else
        log_warning "pip-audit scan found vulnerabilities"
        AUDIT_VULNERABILITIES=$(jq '.vulnerabilities | length' "$audit_output.json" 2>/dev/null || echo "unknown")
    fi
    
    # Human-readable report
    pip-audit --desc --output="$audit_output.txt" 2>&1 || true
    
    echo "$AUDIT_VULNERABILITIES" > "$OUTPUT_DIR/dependency-scans/audit-count.txt"
}

run_bandit_scan() {
    log_info "Running Bandit static security analysis..."
    
    local bandit_output="$OUTPUT_DIR/code-scans/bandit-report"
    
    # JSON report
    bandit -r lightrag/ lightrag_mcp/ -f json -o "$bandit_output.json" -ll || true
    
    # Text report
    bandit -r lightrag/ lightrag_mcp/ -f txt -o "$bandit_output.txt" -ll || true
    
    # Count issues
    if [[ -f "$bandit_output.json" ]]; then
        BANDIT_ISSUES=$(jq '.results | length' "$bandit_output.json" 2>/dev/null || echo "0")
        BANDIT_HIGH=$(jq '[.results[] | select(.issue_severity == "HIGH")] | length' "$bandit_output.json" 2>/dev/null || echo "0")
        BANDIT_MEDIUM=$(jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' "$bandit_output.json" 2>/dev/null || echo "0")
        
        log_info "Bandit found $BANDIT_ISSUES issues (High: $BANDIT_HIGH, Medium: $BANDIT_MEDIUM)"
        
        echo "$BANDIT_ISSUES" > "$OUTPUT_DIR/code-scans/bandit-count.txt"
        echo "$BANDIT_HIGH" > "$OUTPUT_DIR/code-scans/bandit-high-count.txt"
    fi
    
    # Console summary
    echo ""
    log_info "Bandit scan summary:"
    bandit -r lightrag/ lightrag_mcp/ -ll --severity-level medium || true
}

run_semgrep_scan() {
    log_info "Running Semgrep static analysis..."
    
    local semgrep_output="$OUTPUT_DIR/code-scans/semgrep-report"
    
    # Check if semgrep is available
    if ! command -v semgrep &> /dev/null; then
        log_warning "Semgrep not available - skipping advanced static analysis"
        return 0
    fi
    
    # Run Semgrep with security rules
    local configs=(
        "p/security-audit"
        "p/python"
        "p/secrets"
        "p/docker"
    )
    
    for config in "${configs[@]}"; do
        log_info "Running Semgrep with config: $config"
        semgrep --config="$config" \
                --json \
                --output="$semgrep_output-$(basename $config).json" \
                lightrag/ lightrag_mcp/ || true
    done
    
    # Combine results
    if command -v jq &> /dev/null; then
        jq -s 'add' "$semgrep_output"-*.json > "$semgrep_output-combined.json" 2>/dev/null || true
    fi
    
    log_success "Semgrep scan completed"
}

run_secrets_scan() {
    log_info "Running secrets detection..."
    
    local secrets_output="$OUTPUT_DIR/secrets-scans"
    
    # Check for common secret patterns
    log_info "Scanning for hardcoded secrets and credentials..."
    
    # Define secret patterns
    local patterns=(
        "password\s*=\s*['\"][^'\"]*['\"]"
        "api_key\s*=\s*['\"][^'\"]*['\"]"
        "secret\s*=\s*['\"][^'\"]*['\"]"
        "token\s*=\s*['\"][^'\"]*['\"]"
        "AKIA[0-9A-Z]{16}"  # AWS Access Key
        "sk-[a-zA-Z0-9]{32,}"  # OpenAI API Key pattern
        "ghp_[A-Za-z0-9]{36}"  # GitHub Personal Access Token
    )
    
    local secrets_found=0
    
    for pattern in "${patterns[@]}"; do
        log_info "Checking pattern: $pattern"
        
        local matches=$(grep -r -n -E "$pattern" \
                       lightrag/ lightrag_mcp/ \
                       --exclude-dir=__pycache__ \
                       --exclude="*.pyc" \
                       --exclude="test_*" \
                       --exclude="*test*" \
                       2>/dev/null || true)
        
        if [[ -n "$matches" ]]; then
            echo "Pattern: $pattern" >> "$secrets_output/potential-secrets.txt"
            echo "$matches" >> "$secrets_output/potential-secrets.txt"
            echo "---" >> "$secrets_output/potential-secrets.txt"
            secrets_found=$((secrets_found + $(echo "$matches" | wc -l)))
        fi
    done
    
    # Check environment files
    log_info "Checking environment files for potential secrets..."
    
    local env_files=(".env" ".env.local" ".env.example" "env.example")
    for env_file in "${env_files[@]}"; do
        if [[ -f "$env_file" ]]; then
            log_info "Analyzing $env_file..."
            
            # Check for actual values in .env files (not examples)
            if [[ "$env_file" != *"example"* ]]; then
                local env_secrets=$(grep -v "^#" "$env_file" | grep -E "=.+" | grep -v "=your_" | grep -v "=example" | grep -v "=changeme" || true)
                if [[ -n "$env_secrets" ]]; then
                    echo "File: $env_file" >> "$secrets_output/env-analysis.txt"
                    echo "$env_secrets" >> "$secrets_output/env-analysis.txt"
                    echo "---" >> "$secrets_output/env-analysis.txt"
                fi
            fi
        fi
    done
    
    echo "$secrets_found" > "$secrets_output/secrets-count.txt"
    
    if [[ $secrets_found -gt 0 ]]; then
        log_warning "Found $secrets_found potential secrets - review $secrets_output/potential-secrets.txt"
    else
        log_success "No hardcoded secrets detected"
    fi
}

run_container_security_scan() {
    log_info "Running container security analysis..."
    
    local container_output="$OUTPUT_DIR/container-scans"
    
    # Check Dockerfile security
    log_info "Analyzing Dockerfile security..."
    
    local dockerfiles=("Dockerfile" "Dockerfile.production")
    
    for dockerfile in "${dockerfiles[@]}"; do
        if [[ -f "$dockerfile" ]]; then
            log_info "Analyzing $dockerfile..."
            
            # Basic Dockerfile security checks
            python3 << EOF > "$container_output/${dockerfile}-analysis.txt"
import re

dockerfile_path = "$dockerfile"
issues = []

with open(dockerfile_path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    line = line.strip()
    
    # Check for running as root
    if line.startswith('USER root'):
        issues.append(f"Line {i}: Running as root user")
    
    # Check for ADD instead of COPY
    if line.startswith('ADD ') and not ('http' in line or '.tar' in line):
        issues.append(f"Line {i}: Use COPY instead of ADD for local files")
    
    # Check for missing USER directive
    if line.startswith('FROM '):
        user_found = any('USER ' in l for l in lines[i:])
        if not user_found:
            issues.append(f"Line {i}: No USER directive found - container may run as root")
    
    # Check for secrets in ENV
    if line.startswith('ENV ') and any(secret in line.lower() for secret in ['password', 'secret', 'key', 'token']):
        issues.append(f"Line {i}: Potential secret in ENV directive")
    
    # Check for latest tag
    if 'FROM ' in line and ':latest' in line:
        issues.append(f"Line {i}: Using 'latest' tag - use specific versions")

if issues:
    print(f"Security issues found in {dockerfile_path}:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"No security issues found in {dockerfile_path}")
EOF
        fi
    done
    
    # Check for external security tools
    if command -v hadolint &> /dev/null; then
        log_info "Running Hadolint Dockerfile linter..."
        for dockerfile in "${dockerfiles[@]}"; do
            if [[ -f "$dockerfile" ]]; then
                hadolint "$dockerfile" > "$container_output/${dockerfile}-hadolint.txt" 2>&1 || true
            fi
        done
    else
        log_warning "Hadolint not available - install with: brew install hadolint"
    fi
}

analyze_scan_results() {
    log_info "Analyzing security scan results..."
    
    # Count total issues
    local total_vulnerabilities=0
    local high_severity_issues=0
    
    # Dependency vulnerabilities
    if [[ -f "$OUTPUT_DIR/dependency-scans/safety-count.txt" ]]; then
        local safety_count=$(cat "$OUTPUT_DIR/dependency-scans/safety-count.txt")
        total_vulnerabilities=$((total_vulnerabilities + safety_count))
        high_severity_issues=$((high_severity_issues + safety_count))  # All Safety issues are high
    fi
    
    if [[ -f "$OUTPUT_DIR/dependency-scans/audit-count.txt" ]]; then
        local audit_count=$(cat "$OUTPUT_DIR/dependency-scans/audit-count.txt")
        total_vulnerabilities=$((total_vulnerabilities + audit_count))
    fi
    
    # Code security issues
    if [[ -f "$OUTPUT_DIR/code-scans/bandit-high-count.txt" ]]; then
        local bandit_high=$(cat "$OUTPUT_DIR/code-scans/bandit-high-count.txt")
        high_severity_issues=$((high_severity_issues + bandit_high))
    fi
    
    # Secrets
    if [[ -f "$OUTPUT_DIR/secrets-scans/secrets-count.txt" ]]; then
        local secrets_count=$(cat "$OUTPUT_DIR/secrets-scans/secrets-count.txt")
        high_severity_issues=$((high_severity_issues + secrets_count))
    fi
    
    echo "$total_vulnerabilities" > "$OUTPUT_DIR/total-issues.txt"
    echo "$high_severity_issues" > "$OUTPUT_DIR/high-severity-issues.txt"
    
    log_info "Security scan summary:"
    log_info "  Total vulnerabilities: $total_vulnerabilities"
    log_info "  High severity issues: $high_severity_issues"
    
    return $high_severity_issues
}

generate_security_report() {
    log_info "Generating security report..."
    
    local report_file="$OUTPUT_DIR/security-summary.md"
    
    cat > "$report_file" << EOF
# Security Audit Report

**Generated:** $(date -u)
**Project:** LightRAG
**Scan Type:** $SCAN_TYPE

## Executive Summary

$(
if [[ -f "$OUTPUT_DIR/high-severity-issues.txt" ]]; then
    high_issues=$(cat "$OUTPUT_DIR/high-severity-issues.txt")
    if [[ $high_issues -eq 0 ]]; then
        echo "✅ **No high-severity security issues found**"
    else
        echo "⚠️ **$high_issues high-severity security issues found**"
    fi
else
    echo "📊 Security scan completed"
fi
)

## Scan Results

### 📦 Dependency Vulnerabilities

$(
if [[ -f "$OUTPUT_DIR/dependency-scans/safety-count.txt" ]]; then
    safety_count=$(cat "$OUTPUT_DIR/dependency-scans/safety-count.txt")
    echo "- **Safety (PyUp):** $safety_count vulnerabilities"
else
    echo "- **Safety (PyUp):** Not run"
fi

if [[ -f "$OUTPUT_DIR/dependency-scans/audit-count.txt" ]]; then
    audit_count=$(cat "$OUTPUT_DIR/dependency-scans/audit-count.txt")
    echo "- **pip-audit:** $audit_count vulnerabilities"
else
    echo "- **pip-audit:** Not run"
fi
)

### 🔍 Static Code Analysis

$(
if [[ -f "$OUTPUT_DIR/code-scans/bandit-count.txt" ]]; then
    bandit_count=$(cat "$OUTPUT_DIR/code-scans/bandit-count.txt")
    bandit_high=$(cat "$OUTPUT_DIR/code-scans/bandit-high-count.txt" 2>/dev/null || echo "0")
    echo "- **Bandit:** $bandit_count issues ($bandit_high high severity)"
else
    echo "- **Bandit:** Not run"
fi

if [[ -d "$OUTPUT_DIR/code-scans" ]] && ls "$OUTPUT_DIR/code-scans"/semgrep-*.json &>/dev/null; then
    echo "- **Semgrep:** Static analysis completed"
else
    echo "- **Semgrep:** Not available"
fi
)

### 🔐 Secrets Detection

$(
if [[ -f "$OUTPUT_DIR/secrets-scans/secrets-count.txt" ]]; then
    secrets_count=$(cat "$OUTPUT_DIR/secrets-scans/secrets-count.txt")
    echo "- **Secrets scan:** $secrets_count potential secrets found"
else
    echo "- **Secrets scan:** Not run"
fi
)

### 🐳 Container Security

$(
if [[ -d "$OUTPUT_DIR/container-scans" ]]; then
    echo "- **Container analysis:** Completed"
    if command -v hadolint &> /dev/null; then
        echo "- **Hadolint:** Dockerfile linting completed"
    fi
else
    echo "- **Container analysis:** Not run"
fi
)

## Detailed Reports

### Dependency Reports
$(
if [[ -f "$OUTPUT_DIR/dependency-scans/safety-report.json" ]]; then
    echo "- [Safety JSON Report]($OUTPUT_DIR/dependency-scans/safety-report.json)"
    echo "- [Safety Text Report]($OUTPUT_DIR/dependency-scans/safety-report.txt)"
fi

if [[ -f "$OUTPUT_DIR/dependency-scans/pip-audit-report.json" ]]; then
    echo "- [pip-audit JSON Report]($OUTPUT_DIR/dependency-scans/pip-audit-report.json)"
    echo "- [pip-audit Text Report]($OUTPUT_DIR/dependency-scans/pip-audit-report.txt)"
fi
)

### Code Analysis Reports
$(
if [[ -f "$OUTPUT_DIR/code-scans/bandit-report.json" ]]; then
    echo "- [Bandit JSON Report]($OUTPUT_DIR/code-scans/bandit-report.json)"
    echo "- [Bandit Text Report]($OUTPUT_DIR/code-scans/bandit-report.txt)"
fi
)

### Secrets Reports
$(
if [[ -f "$OUTPUT_DIR/secrets-scans/potential-secrets.txt" ]]; then
    echo "- [Potential Secrets]($OUTPUT_DIR/secrets-scans/potential-secrets.txt)"
fi
)

## Recommendations

1. **Immediate Actions:**
   - Review and address all high-severity vulnerabilities
   - Update vulnerable dependencies
   - Rotate any exposed secrets

2. **Security Practices:**
   - Regular dependency updates
   - Implement secrets management
   - Use specific Docker image tags
   - Regular security audits

3. **Monitoring:**
   - Set up automated vulnerability scanning
   - Monitor for new security advisories
   - Implement security alerting

## Commands to Reproduce

\`\`\`bash
# Run full security audit
./scripts/ci/security-audit.sh

# Run specific scans
SCAN_TYPE=dependencies ./scripts/ci/security-audit.sh
SCAN_TYPE=code ./scripts/ci/security-audit.sh
SCAN_TYPE=secrets ./scripts/ci/security-audit.sh
\`\`\`
EOF

    log_success "Security report generated: $report_file"
}

main() {
    log_info "Starting security audit..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Scan type: $SCAN_TYPE"
    log_info "Output directory: $OUTPUT_DIR"
    
    check_dependencies
    
    cd "$PROJECT_ROOT"
    setup_output_directory
    
    # Run scans based on type
    case $SCAN_TYPE in
        "all"|"dependencies")
            run_safety_scan
            run_pip_audit_scan
            ;;& # Continue to next case
        "all"|"code")
            run_bandit_scan
            run_semgrep_scan
            ;;& # Continue to next case
        "all"|"secrets")
            run_secrets_scan
            ;;& # Continue to next case
        "all"|"container")
            run_container_security_scan
            ;;
        *)
            log_error "Invalid scan type: $SCAN_TYPE"
            log_info "Valid types: all, dependencies, code, secrets, container"
            exit 1
            ;;
    esac
    
    # Analyze results
    analyze_scan_results
    high_severity_count=$?
    
    generate_security_report
    
    # Final decision
    if [[ "$FAIL_ON_HIGH" == "true" ]] && [[ $high_severity_count -gt 0 ]]; then
        log_error "Security audit failed: $high_severity_count high-severity issues found"
        log_info "Review the security report: $OUTPUT_DIR/security-summary.md"
        exit 1
    else
        log_success "Security audit completed"
        if [[ $high_severity_count -gt 0 ]]; then
            log_warning "$high_severity_count high-severity issues found but not failing (FAIL_ON_HIGH=false)"
        fi
        exit 0
    fi
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --type TYPE           Scan type (all, dependencies, code, secrets, container)
    -o, --output DIR          Output directory (default: security-reports)
    -f, --fail-on-high        Fail on high-severity issues (default: true)
    -v, --verbose             Verbose output
    -h, --help                Show this help message

Environment Variables:
    SCAN_TYPE                Scan type (default: all)
    OUTPUT_DIR              Output directory (default: security-reports)
    FAIL_ON_HIGH            Fail on high-severity issues (default: true)
    VERBOSE                 Verbose output (default: false)

Examples:
    $0                       # Run all scans
    $0 -t dependencies       # Run only dependency scans
    $0 -t code -o /tmp/sec   # Run code scans with custom output
    FAIL_ON_HIGH=false $0    # Don't fail on high-severity issues
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            SCAN_TYPE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--fail-on-high)
            FAIL_ON_HIGH=true
            shift
            ;;
        --no-fail-on-high)
            FAIL_ON_HIGH=false
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