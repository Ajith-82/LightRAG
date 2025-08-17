#!/bin/bash
set -e

# PostgreSQL SSL Setup for LightRAG Production
# This script configures SSL encryption for PostgreSQL

echo "🔒 Configuring PostgreSQL SSL encryption..."

# Check if SSL certificates exist
SSL_CERT_FILE="/var/lib/postgresql/data/server.crt"
SSL_KEY_FILE="/var/lib/postgresql/data/server.key"
SSL_CA_FILE="/var/lib/postgresql/data/ca.crt"

if [ ! -f "$SSL_CERT_FILE" ] || [ ! -f "$SSL_KEY_FILE" ] || [ ! -f "$SSL_CA_FILE" ]; then
    echo "⚠️  SSL certificates not found. Generating self-signed certificates..."
    
    # Run SSL certificate generation
    if [ -f "/var/lib/postgresql/ssl/generate-ssl-certs.sh" ]; then
        bash /var/lib/postgresql/ssl/generate-ssl-certs.sh
    else
        echo "❌ SSL certificate generation script not found"
        echo "   Continuing without SSL encryption..."
        exit 0
    fi
fi

# Verify certificate files exist and have correct permissions
if [ -f "$SSL_CERT_FILE" ] && [ -f "$SSL_KEY_FILE" ]; then
    echo "✅ SSL certificates found and configured"
    
    # Ensure proper ownership and permissions
    chown 2001:2001 "$SSL_CERT_FILE" "$SSL_KEY_FILE" "$SSL_CA_FILE"
    chmod 644 "$SSL_CERT_FILE" "$SSL_CA_FILE"
    chmod 600 "$SSL_KEY_FILE"
    
    echo "🔒 SSL encryption enabled for PostgreSQL"
    echo "   - Server certificate: $SSL_CERT_FILE"
    echo "   - Server private key: $SSL_KEY_FILE"
    echo "   - CA certificate: $SSL_CA_FILE"
else
    echo "⚠️  SSL certificates could not be generated"
    echo "   PostgreSQL will start without SSL encryption"
fi

# Configure pg_hba.conf for SSL connections
PG_HBA_FILE="/var/lib/postgresql/data/pg_hba.conf"

if [ -f "$PG_HBA_FILE" ]; then
    echo "🔧 Configuring host-based authentication for SSL..."
    
    # Backup original pg_hba.conf
    cp "$PG_HBA_FILE" "${PG_HBA_FILE}.backup"
    
    # Create secure pg_hba.conf that requires SSL
    cat > "$PG_HBA_FILE" << 'EOF'
# PostgreSQL Host-Based Authentication (HBA) Configuration
# Enhanced for LightRAG Production Security

# TYPE  DATABASE        USER            ADDRESS                 METHOD              OPTIONS

# Local connections (Unix domain sockets)
local   all             postgres                                peer
local   all             all                                     scram-sha-256

# IPv4 connections - REQUIRE SSL for security
hostssl all             all             172.20.0.0/16           scram-sha-256
hostssl all             all             127.0.0.1/32            scram-sha-256

# IPv6 connections - REQUIRE SSL for security  
hostssl all             all             ::1/128                 scram-sha-256

# Reject all non-SSL connections for security
host    all             all             all                     reject

# Replication connections (for backup/standby servers)
hostssl replication     postgres        172.20.0.0/16           scram-sha-256
hostssl replication     postgres        127.0.0.1/32            scram-sha-256
EOF

    chown 2001:2001 "$PG_HBA_FILE"
    chmod 600 "$PG_HBA_FILE"
    
    echo "✅ Host-based authentication configured to require SSL"
else
    echo "⚠️  pg_hba.conf not found - SSL authentication rules not configured"
fi

echo "🔒 PostgreSQL SSL setup completed successfully!"