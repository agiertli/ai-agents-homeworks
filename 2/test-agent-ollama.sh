#!/bin/bash

# Test script for the n8n database agent with Ollama

echo "Testing n8n Database Agent with Ollama..."
echo "========================================"

# Test queries
queries=(
    "Show me all products that cost more than 100 dollars"
    "How many customers do we have from Czech Republic?"
    "What is the most expensive product?"
    "List all orders with status pending"
    "What is the total stock quantity of all products?"
)

# Function to send query
send_query() {
    local query="$1"
    echo ""
    echo "Query: $query"
    echo "-------------------"
    
    response=$(curl -s -X POST http://localhost:5678/webhook/database-agent-ollama \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\"}" \
        --max-time 30)
    
    if [ $? -eq 0 ]; then
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        echo "Error: Failed to connect to n8n webhook or request timed out"
    fi
    
    echo ""
    sleep 3
}

# Check if n8n is running
if ! curl -s http://localhost:5678 > /dev/null; then
    echo "Error: n8n is not running on http://localhost:5678"
    echo "Please run ./setup-podman.sh first"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434 > /dev/null; then
    echo "Error: Ollama is not running on http://localhost:11434"
    echo "Please run ./setup-podman.sh first"
    exit 1
fi

# Test Ollama directly
echo "Testing Ollama directly..."
echo "-------------------------"
ollama_test=$(curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistral:7b",
        "prompt": "Say hello in one word",
        "stream": false
    }' \
    --max-time 10)

if [ $? -eq 0 ]; then
    echo "Ollama is responding: OK"
    echo "$ollama_test" | jq '.response' 2>/dev/null || echo "Could not parse response"
else
    echo "Error: Ollama is not responding properly"
fi

echo ""
echo "Starting webhook tests..."
echo "========================"
echo "Note: Ollama responses may be slower than OpenAI"
echo ""

# Send test queries
for query in "${queries[@]}"; do
    send_query "$query"
done

echo "Test completed!"
