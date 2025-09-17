#!/bin/bash

# Setup script for n8n with PostgreSQL using Podman

echo "Setting up n8n with PostgreSQL using Podman..."

# Create a podman network
echo "Creating Podman network..."
podman network create n8n-network 2>/dev/null || echo "Network already exists"

# Create volumes
echo "Creating volumes..."
podman volume create postgres_data 2>/dev/null || echo "postgres_data volume already exists"
podman volume create n8n_data 2>/dev/null || echo "n8n_data volume already exists"
podman volume create ollama_data 2>/dev/null || echo "ollama_data volume already exists"

# Stop and remove existing containers if they exist
echo "Cleaning up existing containers..."
podman stop n8n-postgres n8n ollama 2>/dev/null
podman rm n8n-postgres n8n ollama 2>/dev/null

# Start PostgreSQL container
echo "Starting PostgreSQL container..."
podman run -d \
  --name n8n-postgres \
  --network n8n-network \
  -e POSTGRES_USER=n8n \
  -e POSTGRES_PASSWORD=n8n_password \
  -e POSTGRES_DB=n8n \
  -v postgres_data:/var/lib/postgresql/data \
  -v $(pwd)/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql \
  -p 5432:5432 \
  postgres:15-alpine

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Start Ollama container
echo "Starting Ollama container..."
podman run -d \
  --name ollama \
  --network n8n-network \
  -v ollama_data:/root/.ollama \
  -p 11434:11434 \
  docker.io/ollama/ollama:latest

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 5

# Pull Mistral 7B model
echo "Pulling Mistral 7B model..."
podman exec ollama ollama pull mistral:7b

# Start n8n container
echo "Starting n8n container..."
podman run -d \
  --name n8n \
  --network n8n-network \
  -e DB_TYPE=postgresdb \
  -e DB_POSTGRESDB_HOST=n8n-postgres \
  -e DB_POSTGRESDB_PORT=5432 \
  -e DB_POSTGRESDB_DATABASE=n8n \
  -e DB_POSTGRESDB_USER=n8n \
  -e DB_POSTGRESDB_PASSWORD=n8n_password \
  -e N8N_BASIC_AUTH_ACTIVE=true \
  -e N8N_BASIC_AUTH_USER=admin \
  -e N8N_BASIC_AUTH_PASSWORD=admin \
  -e N8N_HOST=localhost \
  -e N8N_PORT=5678 \
  -e N8N_PROTOCOL=http \
  -e WEBHOOK_URL=http://localhost:5678/ \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  docker.io/n8nio/n8n:latest

echo "Setup complete!"
echo "n8n is available at: http://localhost:5678"
echo "Login credentials: admin/admin"
echo "PostgreSQL is available at: localhost:5432"
echo "Database credentials: n8n/n8n_password"
echo "Ollama API is available at: http://localhost:11434"
echo "Ollama model: mistral:7b"
