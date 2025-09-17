#!/bin/bash

# Test script for the n8n AI Agent with SQL Tool

echo "Testing n8n AI Agent with SQL Tool and Ollama..."
echo "==============================================="

# Test queries
queries=(
    "What products do you have in the Electronics category?"
    "Show me the total number of customers from Czech Republic"
    "What is the average price of all products?"
    "List the 3 most expensive products with their prices"
    "How many orders are currently pending?"
    "What is the total revenue from all delivered orders?"
)

# Function to send query
send_query() {
    local query="$1"
    echo ""
    echo "Query: $query"
    echo "-------------------"
    
    response=$(curl -s -X POST http://localhost:5678/webhook/ai-agent-sql \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\"}" \
        --max-time 45)
    
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

echo "Starting AI Agent tests..."
echo "========================="
echo "Note: The AI Agent may take longer to respond as it reasons through the query"
echo ""

# Send test queries
for query in "${queries[@]}"; do
    send_query "$query"
done

echo "Test completed!"
echo ""
echo "The AI Agent workflow demonstrates:"
echo "- Natural language understanding via Ollama (Mistral 7B)"
echo "- SQL Tool usage for database queries"
echo "- Intelligent query generation and result formatting"
