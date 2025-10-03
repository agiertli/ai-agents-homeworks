# PostgreSQL MCP ReAct Agent

This directory contains a ReAct (Reasoning and Acting) agent implementation using PostgreSQL with Model Context Protocol (MCP) and LangChain.

## What's Here

### Core Files
- **clean_demo.py** - ReAct agent implementation featuring:
  - Full ReAct (Reasoning and Acting) pattern using LangChain's `create_react_agent`
  - MCP server connection (crystaldba/postgres-mcp) for database tools
  - Custom parser for handling JSON inputs
  - Interactive mode for continuous conversations
  - AgentExecutor for managing the reasoning loop

- **clean_demo_simple.py** - Simplified version without ReAct loop:
  - Direct LLM prompting for SQL generation
  - Manual tool execution
  - Good for testing basic MCP connectivity

- **config.py** - Loads configuration from .env file

- **setup-mcp-postgres.sh** - Sets up:
  - PostgreSQL database with sample data
  - crystaldba/postgres-mcp server with SSE transport

### Configuration
- **.env** - Your environment variables:
  ```
  OPENAI_API_KEY=your-key
  OPENAI_BASE_URL=your-llm-endpoint
  MODEL_NAME=your-model-name
  ```

## Quick Start

1. Start the database and MCP server:
   ```bash
   ./setup-mcp-postgres.sh
   ```

2. Run the demo:
   ```bash
   source venv/bin/activate
   python clean_demo.py                    # Run single query demo
   python clean_demo.py --interactive      # Interactive ReAct agent mode
   python clean_demo_simple.py             # Simple version without ReAct
   ```

## Key Concepts

### ReAct Pattern
This implementation uses the ReAct (Reasoning and Acting) pattern:
1. **Thought**: The agent reasons about what to do next
2. **Action**: The agent selects a tool to use
3. **Action Input**: The agent provides input for the tool
4. **Observation**: The agent observes the tool's output
5. **Loop**: Steps 1-4 repeat until the agent has enough information
6. **Final Answer**: The agent provides the final response

### Components
- **MCP (Model Context Protocol)**: Provides database tools that the agent can use
- **ReAct Agent**: Implements the reasoning loop using LangChain's `create_react_agent`
- **AgentExecutor**: Manages the iterative process and tool execution
- **LLM**: Powers the reasoning and decision-making

### Why Not LangGraph?
This uses LangChain's built-in ReAct implementation rather than LangGraph because:
- It's simpler and more straightforward for basic agents
- LangGraph is better suited for complex workflows with branching logic
- For database queries, the linear ReAct pattern is sufficient

## Current Status

- ✅ MCP server connection works
- ✅ Tools are loaded and available to the agent
- ✅ ReAct agent implementation with proper prompt template
- ✅ Custom parser handles various JSON input formats
- ✅ Interactive mode for continuous conversations
- ✅ Direct tool execution works
- ✅ Agent successfully follows the ReAct pattern when working properly

## Example ReAct Flow

When you ask "Show me all users over 25 years old", the agent follows:
```
Question: Show me all users over 25 years old
Thought: I need to query the database to find users with age > 25
Action: execute_sql
Action Input: {"sql": "SELECT * FROM users WHERE age > 25"}
Observation: [query results]
Thought: I now have the user data
Final Answer: Here are the users over 25 years old: [formatted results]
```

## Troubleshooting

If the agent doesn't generate proper tool calls:
- Ensure your LLM supports the ReAct format and can follow structured prompts
- The custom parser helps handle various JSON input formats
- Check that the MCP server is running and accessible
- Verify your `.env` configuration is correct
