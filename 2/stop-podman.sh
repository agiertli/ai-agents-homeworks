#!/bin/bash

# Stop script for n8n and PostgreSQL containers

echo "Stopping n8n, PostgreSQL, and Ollama containers..."

podman stop n8n n8n-postgres ollama
podman rm n8n n8n-postgres ollama

echo "Containers stopped and removed."
echo "Note: Volumes are preserved. To remove them, run:"
echo "  podman volume rm postgres_data n8n_data ollama_data"
