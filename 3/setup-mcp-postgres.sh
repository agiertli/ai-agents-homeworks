#!/bin/bash

# Setup script for crystaldba/postgres-mcp server
# This uses the crystaldba PostgreSQL MCP server from Docker Hub

set -e

# Configuration
MCP_CONTAINER_NAME="mcp-postgres-server"
POSTGRES_CONTAINER_NAME="postgres-db"
POSTGRES_DB="testdb"
POSTGRES_USER="testuser"
POSTGRES_PASSWORD="testpass"
POSTGRES_PORT="5432"
MCP_PORT="8000"

echo "🐘 Setting up PostgreSQL database and MCP server with Podman..."
echo "================================================="

# Stop and remove existing containers if they exist
echo "Stopping existing containers (if any)..."
podman stop $POSTGRES_CONTAINER_NAME 2>/dev/null || true
podman rm $POSTGRES_CONTAINER_NAME 2>/dev/null || true
podman stop $MCP_CONTAINER_NAME 2>/dev/null || true
podman rm $MCP_CONTAINER_NAME 2>/dev/null || true

# Create a network for containers to communicate
echo "Creating podman network..."
podman network create mcp-network 2>/dev/null || true

# 1. Setup PostgreSQL database first
echo ""
echo "📦 Setting up PostgreSQL database..."
podman run -d \
  --name $POSTGRES_CONTAINER_NAME \
  --network mcp-network \
  -e POSTGRES_DB=$POSTGRES_DB \
  -e POSTGRES_USER=$POSTGRES_USER \
  -e POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  -p $POSTGRES_PORT:5432 \
  postgres:15

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Test PostgreSQL connection
echo "Testing PostgreSQL connection..."
podman exec $POSTGRES_CONTAINER_NAME pg_isready -U $POSTGRES_USER -d $POSTGRES_DB

# Create sample tables and data
echo "Creating sample database schema and data..."
podman exec -i $POSTGRES_CONTAINER_NAME psql -U $POSTGRES_USER -d $POSTGRES_DB << 'EOF'
-- Create sample tables
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    product_name VARCHAR(200) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO users (name, email, age) VALUES 
    ('Alice Johnson', 'alice@example.com', 28),
    ('Bob Smith', 'bob@example.com', 34),
    ('Carol Davis', 'carol@example.com', 22),
    ('David Wilson', 'david@example.com', 45)
ON CONFLICT (email) DO NOTHING;

INSERT INTO orders (user_id, product_name, quantity, price) VALUES 
    (1, 'Laptop', 1, 1299.99),
    (1, 'Mouse', 2, 25.50),
    (2, 'Keyboard', 1, 89.99),
    (3, 'Monitor', 1, 299.99),
    (4, 'Tablet', 1, 499.99)
ON CONFLICT DO NOTHING;

-- Show tables
\dt
SELECT COUNT(*) as user_count FROM users;
SELECT COUNT(*) as order_count FROM orders;
EOF

# 2. Pull and run the crystaldba/postgres-mcp server
echo ""
echo "📦 Setting up crystaldba/postgres-mcp server..."
echo "Pulling crystaldba/postgres-mcp server image..."
podman pull crystaldba/postgres-mcp:latest

# Run crystaldba/postgres-mcp server
echo "Starting crystaldba/postgres-mcp server..."
DATABASE_URI="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_CONTAINER_NAME:5432/$POSTGRES_DB"
podman run -d \
  --name $MCP_CONTAINER_NAME \
  --network mcp-network \
  -p 8000:8000 \
  -e DATABASE_URI="$DATABASE_URI" \
  crystaldba/postgres-mcp:latest \
  --access-mode=unrestricted \
  --transport=sse

# Wait for MCP server to be ready
echo "Waiting for MCP server to start..."
sleep 5

# Test MCP server
echo "Testing MCP server connection..."
if curl -s http://localhost:$MCP_PORT/health > /dev/null 2>&1; then
    echo "✅ MCP server is running"
else
    echo "⚠️  MCP server health check not available, checking container status..."
    podman ps | grep $MCP_CONTAINER_NAME
fi

echo ""
echo "✅ Setup complete!"
echo "=================="
echo "PostgreSQL Database:"
echo "  Container: $POSTGRES_CONTAINER_NAME"
echo "  Port: $POSTGRES_PORT"
echo "  Database: $POSTGRES_DB"
echo "  User: $POSTGRES_USER"
echo ""
echo "MCP PostgreSQL Server (crystaldba/postgres-mcp):"
echo "  Container: $MCP_CONTAINER_NAME"
echo "  Port: $MCP_PORT"
echo "  SSE Endpoint: http://localhost:$MCP_PORT/sse"
echo "  Transport: SSE (Server-Sent Events)"
echo "  Access Mode: unrestricted"
echo ""
echo "To view logs:"
echo "  podman logs $MCP_CONTAINER_NAME"
echo "  podman logs $POSTGRES_CONTAINER_NAME"
echo ""
echo "To stop services:"
echo "  podman stop $MCP_CONTAINER_NAME $POSTGRES_CONTAINER_NAME"
echo ""
echo "To connect to PostgreSQL directly:"
echo "  podman exec -it $POSTGRES_CONTAINER_NAME psql -U $POSTGRES_USER -d $POSTGRES_DB"

