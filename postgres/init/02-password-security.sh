#!/bin/bash
set -e

# PostgreSQL Password Security Enhancement for LightRAG
# This script enforces strong password policies and validates credentials

echo "PostgreSQL Security: Validating password strength requirements..."

CUSTOM_PASSWORD="${POSTGRES_PASSWORD}"

# Function to validate password strength
validate_password() {
    local password="$1"
    local errors=()

    # Check minimum length (16 characters)
    if [ ${#password} -lt 16 ]; then
        errors+=("Password must be at least 16 characters long (current: ${#password})")
    fi

    # Check for uppercase letters
    if ! echo "$password" | grep -q '[A-Z]'; then
        errors+=("Password must contain uppercase letters")
    fi

    # Check for lowercase letters
    if ! echo "$password" | grep -q '[a-z]'; then
        errors+=("Password must contain lowercase letters")
    fi

    # Check for numbers
    if ! echo "$password" | grep -q '[0-9]'; then
        errors+=("Password must contain numbers")
    fi

    # Check for special characters
    if ! echo "$password" | grep -q '[^A-Za-z0-9]'; then
        errors+=("Password must contain special characters")
    fi

    # Check for common weak passwords
    local weak_patterns=("password" "123456" "qwerty" "admin" "root" "user" "test" "lightrag" "rag")
    for pattern in "${weak_patterns[@]}"; do
        if echo "$password" | grep -qi "$pattern"; then
            errors+=("Password contains weak pattern: $pattern")
        fi
    done

    # Return validation results
    if [ ${#errors[@]} -eq 0 ]; then
        echo "✅ Password validation passed"
        return 0
    else
        echo "❌ Password validation failed:"
        for error in "${errors[@]}"; do
            echo "  - $error"
        done
        return 1
    fi
}

# Check if password is provided
if [ -z "$CUSTOM_PASSWORD" ]; then
    echo "❌ CRITICAL ERROR: POSTGRES_PASSWORD environment variable is not set"
    echo "   Please set a strong password following these requirements:"
    echo "   - At least 16 characters long"
    echo "   - Contains uppercase and lowercase letters"
    echo "   - Contains numbers and special characters"
    echo "   - Does not contain common weak patterns"
    exit 1
fi

# Check for default weak passwords
if [ "$CUSTOM_PASSWORD" = "rag" ] || [ "$CUSTOM_PASSWORD" = "lightrag" ] || [ "$CUSTOM_PASSWORD" = "password" ]; then
    echo "❌ CRITICAL ERROR: Using default weak password detected"
    echo "   Current password: $CUSTOM_PASSWORD"
    echo "   Please change to a strong password meeting security requirements"
    exit 1
fi

# Validate password strength
if ! validate_password "$CUSTOM_PASSWORD"; then
    echo ""
    echo "❌ CRITICAL ERROR: Password does not meet security requirements"
    echo ""
    echo "🔒 SECURITY REQUIREMENTS:"
    echo "   ✓ Minimum 16 characters"
    echo "   ✓ At least one uppercase letter (A-Z)"
    echo "   ✓ At least one lowercase letter (a-z)"
    echo "   ✓ At least one number (0-9)"
    echo "   ✓ At least one special character (!@#$%^&*)"
    echo "   ✓ No common weak patterns"
    echo ""
    echo "💡 Example strong password:"
    echo "   MySecureLightRAG2025!@#"
    echo ""
    exit 1
fi

echo "✅ PostgreSQL password security validation completed successfully"
echo "🔒 Strong password policy enforced for production deployment"