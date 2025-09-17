#!/bin/bash

# Test script for the n8n database agent

echo "Testing n8n Database Agent..."
echo "=============================="

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
    
    response=$(curl -s -X POST http://localhost:5678/webhook/database-agent \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\"}")
    
    if [ $? -eq 0 ]; then
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        echo "Error: Failed to connect to n8n webhook"
    fi
    
    echo ""
    sleep 2
}

# Check if n8n is running
if ! curl -s http://localhost:5678 > /dev/null; then
    echo "Error: n8n is not running on http://localhost:5678"
    echo "Please run ./setup-podman.sh first"
    exit 1
fi

# Send test queries
for query in "${queries[@]}"; do
    send_query "$query"
done

echo "Test completed!"
