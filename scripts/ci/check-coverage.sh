#!/bin/bash
set -euo pipefail

# Coverage threshold enforcement script
# This script enforces coverage thresholds and generates coverage reports

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-70}
COVERAGE_FAIL_UNDER=${COVERAGE_FAIL_UNDER:-$COVERAGE_THRESHOLD}

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
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if ! python3 -c "import coverage" 2>/dev/null; then
        log_error "Coverage package not found. Install with: pip install coverage[toml]"
        exit 1
    fi
    
    log_success "All dependencies found"
}

run_coverage_analysis() {
    log_info "Running coverage analysis..."
    
    cd "$PROJECT_ROOT"
    
    # Check if coverage data exists
    if [[ ! -f .coverage ]] && [[ ! -f coverage.xml ]]; then
        log_warning "No coverage data found. Running tests with coverage..."
        
        # Run tests with coverage
        python -m pytest tests/ \
            --cov=lightrag \
            --cov=lightrag_mcp \
            --cov-report=term-missing \
            --cov-report=html:htmlcov \
            --cov-report=xml:coverage.xml \
            --cov-fail-under="$COVERAGE_FAIL_UNDER" \
            -v
    else
        log_info "Using existing coverage data"
    fi
    
    # Generate reports
    log_info "Generating coverage reports..."
    
    # Terminal report
    coverage report --show-missing
    
    # HTML report
    coverage html --directory htmlcov
    
    # XML report for CI/CD tools
    coverage xml -o coverage.xml
    
    # JSON report for further processing
    coverage json -o coverage.json
}

check_coverage_threshold() {
    log_info "Checking coverage threshold (minimum: ${COVERAGE_THRESHOLD}%)..."
    
    # Get overall coverage percentage
    if [[ -f coverage.json ]]; then
        ACTUAL_COVERAGE=$(python3 -c "
import json
with open('coverage.json', 'r') as f:
    data = json.load(f)
print(f\"{data['totals']['percent_covered']:.2f}\")
")
    else
        # Fallback to parsing coverage report
        ACTUAL_COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    fi
    
    log_info "Actual coverage: ${ACTUAL_COVERAGE}%"
    
    # Compare with threshold
    if (( $(echo "$ACTUAL_COVERAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then
        log_success "Coverage threshold met: ${ACTUAL_COVERAGE}% >= ${COVERAGE_THRESHOLD}%"
        return 0
    else
        log_error "Coverage threshold not met: ${ACTUAL_COVERAGE}% < ${COVERAGE_THRESHOLD}%"
        return 1
    fi
}

check_individual_file_coverage() {
    log_info "Checking individual file coverage..."
    
    # Files that should have high coverage (>80%)
    HIGH_COVERAGE_FILES=(
        "lightrag/lightrag.py"
        "lightrag/base.py"
        "lightrag/utils.py"
    )
    
    # Files that are allowed lower coverage (<50%)
    LOW_COVERAGE_ALLOWED=(
        "lightrag/llm/"
        "lightrag/kg/deprecated/"
        "examples/"
        "tests/"
    )
    
    FAILED_FILES=()
    
    if [[ -f coverage.json ]]; then
        python3 << 'EOF'
import json
import sys

with open('coverage.json', 'r') as f:
    data = json.load(f)

failed_files = []
high_coverage_files = [
    "lightrag/lightrag.py",
    "lightrag/base.py", 
    "lightrag/utils.py"
]

low_coverage_allowed = [
    "lightrag/llm/",
    "lightrag/kg/deprecated/",
    "examples/",
    "tests/"
]

for file_path, file_data in data['files'].items():
    coverage_percent = file_data['summary']['percent_covered']
    
    # Check if file should have high coverage
    is_high_coverage_file = any(hcf in file_path for hcf in high_coverage_files)
    is_low_coverage_allowed = any(lca in file_path for lca in low_coverage_allowed)
    
    if is_high_coverage_file and coverage_percent < 80:
        print(f"❌ {file_path}: {coverage_percent:.1f}% (should be >80%)")
        failed_files.append(file_path)
    elif not is_low_coverage_allowed and coverage_percent < 60:
        print(f"⚠️  {file_path}: {coverage_percent:.1f}% (should be >60%)")
    elif coverage_percent >= 90:
        print(f"✅ {file_path}: {coverage_percent:.1f}%")

if failed_files:
    print(f"\n{len(failed_files)} files failed coverage requirements")
    sys.exit(1)
else:
    print("\n✅ All files meet coverage requirements")
EOF
    fi
}

generate_coverage_badge() {
    log_info "Generating coverage badge..."
    
    if [[ -f coverage.json ]]; then
        COVERAGE_PERCENT=$(python3 -c "
import json
with open('coverage.json', 'r') as f:
    data = json.load(f)
print(int(data['totals']['percent_covered']))
")
        
        # Determine badge color
        if (( COVERAGE_PERCENT >= 90 )); then
            COLOR="brightgreen"
        elif (( COVERAGE_PERCENT >= 80 )); then
            COLOR="green"
        elif (( COVERAGE_PERCENT >= 70 )); then
            COLOR="yellow"
        elif (( COVERAGE_PERCENT >= 60 )); then
            COLOR="orange"
        else
            COLOR="red"
        fi
        
        # Generate badge URL
        BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_PERCENT}%25-${COLOR}"
        
        log_info "Coverage badge: $BADGE_URL"
        echo "$BADGE_URL" > coverage-badge.txt
    fi
}

analyze_coverage_trends() {
    log_info "Analyzing coverage trends..."
    
    # Check if previous coverage data exists
    if [[ -f coverage-history.json ]]; then
        python3 << 'EOF'
import json
import os
from datetime import datetime

# Load current coverage
with open('coverage.json', 'r') as f:
    current_data = json.load(f)

current_coverage = current_data['totals']['percent_covered']

# Load historical data
history_file = 'coverage-history.json'
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        history = json.load(f)
else:
    history = {'entries': []}

# Add current entry
history['entries'].append({
    'timestamp': datetime.now().isoformat(),
    'coverage': current_coverage,
    'commit': os.environ.get('CI_COMMIT_SHA', os.environ.get('GITHUB_SHA', 'unknown'))
})

# Keep only last 50 entries
history['entries'] = history['entries'][-50:]

# Save updated history
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)

# Analyze trend
if len(history['entries']) >= 2:
    previous_coverage = history['entries'][-2]['coverage']
    change = current_coverage - previous_coverage
    
    if change > 0:
        print(f"📈 Coverage improved by {change:.2f}% (from {previous_coverage:.2f}% to {current_coverage:.2f}%)")
    elif change < 0:
        print(f"📉 Coverage decreased by {abs(change):.2f}% (from {previous_coverage:.2f}% to {current_coverage:.2f}%)")
    else:
        print(f"➡️  Coverage unchanged at {current_coverage:.2f}%")
else:
    print(f"📊 Initial coverage measurement: {current_coverage:.2f}%")
EOF
    else
        log_info "No previous coverage data found. Starting coverage tracking."
    fi
}

generate_coverage_summary() {
    log_info "Generating coverage summary..."
    
    cat > coverage-summary.md << EOF
# Coverage Report Summary

**Generated:** $(date -u)
**Threshold:** ${COVERAGE_THRESHOLD}%
**Actual Coverage:** ${ACTUAL_COVERAGE}%

## Status

$(if (( $(echo "$ACTUAL_COVERAGE >= $COVERAGE_THRESHOLD" | bc -l) )); then echo "✅ **PASSED** - Coverage threshold met"; else echo "❌ **FAILED** - Coverage below threshold"; fi)

## Files

### High Coverage (>90%)
$(coverage report | awk 'NR>2 && $(NF-1)+0 >= 90 {print "- " $1 ": " $(NF-1)}' || echo "None")

### Low Coverage (<60%)
$(coverage report | awk 'NR>2 && $(NF-1)+0 < 60 {print "- " $1 ": " $(NF-1)}' || echo "None")

## Reports

- [HTML Report](htmlcov/index.html)
- [XML Report](coverage.xml)
- [JSON Report](coverage.json)

## Commands

To run coverage locally:
\`\`\`bash
python -m pytest tests/ --cov=lightrag --cov=lightrag_mcp --cov-report=html
\`\`\`

To check specific coverage:
\`\`\`bash
coverage report --show-missing
\`\`\`
EOF

    log_success "Coverage summary generated: coverage-summary.md"
}

main() {
    log_info "Starting coverage analysis..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Coverage threshold: $COVERAGE_THRESHOLD%"
    
    check_dependencies
    
    cd "$PROJECT_ROOT"
    
    run_coverage_analysis
    
    # Check thresholds
    THRESHOLD_PASSED=true
    if ! check_coverage_threshold; then
        THRESHOLD_PASSED=false
    fi
    
    # Additional checks
    check_individual_file_coverage || true
    generate_coverage_badge
    analyze_coverage_trends
    
    # Set ACTUAL_COVERAGE for summary
    if [[ -f coverage.json ]]; then
        ACTUAL_COVERAGE=$(python3 -c "
import json
with open('coverage.json', 'r') as f:
    data = json.load(f)
print(f\"{data['totals']['percent_covered']:.2f}\")
")
    fi
    
    generate_coverage_summary
    
    if [[ "$THRESHOLD_PASSED" == "true" ]]; then
        log_success "Coverage analysis completed successfully"
        exit 0
    else
        log_error "Coverage analysis failed - threshold not met"
        exit 1
    fi
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --threshold PERCENT    Set coverage threshold (default: $COVERAGE_THRESHOLD)
    -h, --help                 Show this help message

Environment Variables:
    COVERAGE_THRESHOLD         Coverage threshold percentage (default: 70)
    COVERAGE_FAIL_UNDER       Fail threshold for pytest-cov (default: same as threshold)

Examples:
    $0                         # Use default threshold (70%)
    $0 -t 80                   # Set threshold to 80%
    COVERAGE_THRESHOLD=85 $0   # Set threshold via environment variable
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threshold)
            COVERAGE_THRESHOLD="$2"
            COVERAGE_FAIL_UNDER="$2"
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