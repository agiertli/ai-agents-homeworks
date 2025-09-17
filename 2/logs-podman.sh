#!/bin/bash

# Script to view logs from n8n and PostgreSQL containers

if [ "$1" == "n8n" ]; then
    echo "Showing n8n logs..."
    podman logs -f n8n
elif [ "$1" == "postgres" ]; then
    echo "Showing PostgreSQL logs..."
    podman logs -f n8n-postgres
elif [ "$1" == "ollama" ]; then
    echo "Showing Ollama logs..."
    podman logs -f ollama
else
    echo "Usage: ./logs-podman.sh [n8n|postgres|ollama]"
    echo ""
    echo "Available containers:"
    echo "  n8n      - Show n8n application logs"
    echo "  postgres - Show PostgreSQL database logs"
    echo "  ollama   - Show Ollama LLM server logs"
fi
