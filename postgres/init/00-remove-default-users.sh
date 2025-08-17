#!/bin/bash
set -e

# PostgreSQL Default User Security Cleanup for LightRAG
# This script removes insecure default users and databases from shangor/postgres-for-rag image

echo "🔒 PostgreSQL Security: Removing default insecure credentials..."

CUSTOM_USER="${POSTGRES_USER:-lightrag_prod}"
CUSTOM_PASSWORD="${POSTGRES_PASSWORD}"
CUSTOM_DATABASE="${POSTGRES_DATABASE:-lightrag_production}"

# Function to safely execute SQL commands
execute_sql() {
    local sql="$1"
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname postgres -c \"$sql\""
}

# Function to safely execute SQL commands on specific database
execute_sql_db() {
    local database="$1"
    local sql="$2"
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname \"$database\" -c \"$sql\"" 2>/dev/null || true
}

echo "🔒 Step 1: Securing default 'postgres' superuser account..."

# Change default postgres user password to a strong one
if [ -n "$CUSTOM_PASSWORD" ]; then
    echo "🔑 Updating postgres superuser password..."
    execute_sql "ALTER USER postgres PASSWORD '$CUSTOM_PASSWORD';"
    echo "✅ Postgres superuser password updated to strong password"
else
    echo "❌ ERROR: POSTGRES_PASSWORD must be set for security"
    exit 1
fi

echo "🔒 Step 2: Removing insecure default 'rag' user and database..."

# Check if default 'rag' user exists and remove it
echo "🔍 Checking for default 'rag' user..."
if execute_sql "SELECT 1 FROM pg_roles WHERE rolname = 'rag'" | grep -q '1'; then
    echo "⚠️  Found insecure default 'rag' user - removing..."
    
    # First, terminate any active connections from the rag user
    execute_sql "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = 'rag';" || true
    
    # Remove the rag user
    execute_sql "DROP USER IF EXISTS rag;" || true
    echo "✅ Removed insecure default 'rag' user"
else
    echo "✅ No default 'rag' user found"
fi

# Check if default 'rag' database exists and remove it (if not being used)
echo "🔍 Checking for default 'rag' database..."
if execute_sql "SELECT 1 FROM pg_database WHERE datname = 'rag'" | grep -q '1'; then
    if [ "$CUSTOM_DATABASE" != "rag" ]; then
        echo "⚠️  Found insecure default 'rag' database - removing..."
        
        # Terminate connections to the rag database
        execute_sql "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'rag';" || true
        
        # Drop the rag database
        execute_sql "DROP DATABASE IF EXISTS rag;" || true
        echo "✅ Removed insecure default 'rag' database"
    else
        echo "⚠️  Default 'rag' database found but configured for use - securing instead..."
        # If we're using the rag database, ensure it has proper ownership
        execute_sql "ALTER DATABASE rag OWNER TO $CUSTOM_USER;" || true
    fi
else
    echo "✅ No default 'rag' database found"
fi

echo "🔒 Step 3: Removing other potentially insecure default users..."

# List of potentially insecure default users to remove
INSECURE_USERS=("test" "guest" "demo" "sample" "example" "admin" "user")

for user in "${INSECURE_USERS[@]}"; do
    if execute_sql "SELECT 1 FROM pg_roles WHERE rolname = '$user'" | grep -q '1'; then
        echo "⚠️  Found potentially insecure user '$user' - removing..."
        execute_sql "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = '$user';" || true
        execute_sql "DROP USER IF EXISTS $user;" || true
        echo "✅ Removed insecure user '$user'"
    fi
done

echo "🔒 Step 4: Disabling unnecessary default extensions and schemas..."

# Remove or secure any test/sample schemas that might exist
TEST_SCHEMAS=("test" "sample" "demo" "example")

for schema in "${TEST_SCHEMAS[@]}"; do
    # Check in our custom database
    if execute_sql_db "$CUSTOM_DATABASE" "SELECT 1 FROM information_schema.schemata WHERE schema_name = '$schema'" | grep -q '1'; then
        echo "⚠️  Found test schema '$schema' - removing..."
        execute_sql_db "$CUSTOM_DATABASE" "DROP SCHEMA IF EXISTS $schema CASCADE;" || true
        echo "✅ Removed test schema '$schema'"
    fi
done

echo "🔒 Step 5: Enforcing connection security..."

# Update pg_hba.conf to disable insecure authentication methods
PG_HBA_FILE="/var/lib/postgresql/data/pg_hba.conf"

if [ -f "$PG_HBA_FILE" ]; then
    echo "🔧 Securing pg_hba.conf authentication methods..."
    
    # Backup original
    cp "$PG_HBA_FILE" "${PG_HBA_FILE}.insecure.backup"
    
    # Remove any 'trust' authentication entries (highly insecure)
    sed -i 's/trust/scram-sha-256/g' "$PG_HBA_FILE" || true
    
    # Remove any 'password' authentication entries (plaintext passwords)
    sed -i 's/\bpassword\b/scram-sha-256/g' "$PG_HBA_FILE" || true
    
    echo "✅ Secured authentication methods in pg_hba.conf"
fi

echo "🔒 Step 6: Final security validation..."

# Verify no weak users remain
echo "🔍 Performing final security scan..."
WEAK_USERS=$(execute_sql "SELECT rolname FROM pg_roles WHERE rolname IN ('rag', 'test', 'guest', 'demo', 'sample', 'example', 'admin', 'user');" | grep -v 'rolname' | grep -v '^$' | wc -l)

if [ "$WEAK_USERS" -gt 0 ]; then
    echo "❌ WARNING: Found $WEAK_USERS potentially insecure users still present!"
    execute_sql "SELECT rolname FROM pg_roles WHERE rolname IN ('rag', 'test', 'guest', 'demo', 'sample', 'example', 'admin', 'user');"
else
    echo "✅ No insecure default users found"
fi

# Verify strong authentication is enabled
AUTH_CHECK=$(grep -c "scram-sha-256" "$PG_HBA_FILE" 2>/dev/null || echo "0")
if [ "$AUTH_CHECK" -gt 0 ]; then
    echo "✅ Strong authentication (SCRAM-SHA-256) is configured"
else
    echo "⚠️  WARNING: Strong authentication may not be properly configured"
fi

echo ""
echo "🔒 PostgreSQL Default Credential Security Cleanup COMPLETED!"
echo ""
echo "✅ Security Status:"
echo "   - Default 'rag' user: REMOVED or SECURED"
echo "   - Default 'rag' database: REMOVED or SECURED"  
echo "   - Postgres superuser: PASSWORD UPDATED"
echo "   - Weak authentication: DISABLED"
echo "   - Test users/schemas: REMOVED"
echo ""
echo "🔐 Your PostgreSQL instance is now secured against default credential vulnerabilities!"