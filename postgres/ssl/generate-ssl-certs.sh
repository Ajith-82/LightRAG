#!/bin/bash
set -e

# PostgreSQL SSL Certificate Generation for LightRAG Production
# This script generates self-signed SSL certificates for PostgreSQL encryption

CERT_DIR="/var/lib/postgresql/ssl"
DATA_DIR="/var/lib/postgresql/data"

echo "🔒 Generating SSL certificates for PostgreSQL..."

# Create certificate directory
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# Generate CA private key
echo "📋 Generating Certificate Authority (CA) private key..."
openssl genrsa -out ca.key 4096
chmod 600 ca.key
chown 2001:2001 ca.key

# Generate CA certificate
echo "📋 Generating Certificate Authority (CA) certificate..."
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt -subj "/C=US/ST=Production/L=LightRAG/O=LightRAG/OU=Database/CN=LightRAG-CA"
chmod 644 ca.crt
chown 2001:2001 ca.crt

# Generate server private key
echo "📋 Generating PostgreSQL server private key..."
openssl genrsa -out server.key 4096
chmod 600 server.key
chown 2001:2001 server.key

# Generate server certificate signing request
echo "📋 Generating PostgreSQL server certificate signing request..."
openssl req -new -key server.key -out server.csr -subj "/C=US/ST=Production/L=LightRAG/O=LightRAG/OU=Database/CN=postgres"

# Create extensions file for server certificate
cat > server.ext << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = postgres
DNS.2 = lightrag_postgres
DNS.3 = lightrag_postgres_enhanced
DNS.4 = postgres-enhanced
DNS.5 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Generate server certificate signed by CA
echo "📋 Generating PostgreSQL server certificate..."
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -extensions v3_req -extfile server.ext
chmod 644 server.crt
chown 2001:2001 server.crt

# Generate client private key
echo "📋 Generating client private key..."
openssl genrsa -out client.key 4096
chmod 600 client.key
chown 2001:2001 client.key

# Generate client certificate signing request
echo "📋 Generating client certificate signing request..."
openssl req -new -key client.key -out client.csr -subj "/C=US/ST=Production/L=LightRAG/O=LightRAG/OU=Client/CN=lightrag_client"

# Generate client certificate signed by CA
echo "📋 Generating client certificate..."
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 365
chmod 644 client.crt
chown 2001:2001 client.crt

# Copy certificates to PostgreSQL data directory
echo "📋 Installing certificates in PostgreSQL data directory..."
cp ca.crt "$DATA_DIR/"
cp server.crt "$DATA_DIR/"
cp server.key "$DATA_DIR/"
cp client.crt "$DATA_DIR/"
cp client.key "$DATA_DIR/"

# Set proper permissions
chown 2001:2001 "$DATA_DIR"/*.crt "$DATA_DIR"/*.key
chmod 644 "$DATA_DIR"/*.crt
chmod 600 "$DATA_DIR"/*.key

# Clean up temporary files
rm -f server.csr client.csr server.ext ca.srl

echo "✅ SSL certificates generated successfully!"
echo ""
echo "📁 Certificate files created:"
echo "   - CA Certificate: $DATA_DIR/ca.crt"
echo "   - Server Certificate: $DATA_DIR/server.crt"
echo "   - Server Private Key: $DATA_DIR/server.key"
echo "   - Client Certificate: $DATA_DIR/client.crt"
echo "   - Client Private Key: $DATA_DIR/client.key"
echo ""
echo "🔒 PostgreSQL SSL encryption is now configured!"
echo ""
echo "📋 To connect with SSL, use connection string:"
echo "   postgresql://user:password@postgres:5432/database?sslmode=require"
echo ""
echo "🔐 For client certificate authentication, use:"
echo "   postgresql://user:password@postgres:5432/database?sslmode=require&sslcert=client.crt&sslkey=client.key&sslrootcert=ca.crt"