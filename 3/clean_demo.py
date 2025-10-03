#!/usr/bin/env python3
"""
Clean demo showing exactly how MCP tools work with LLM
"""
import asyncio
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re
import json as json_module

# Load environment variables
load_dotenv()

class CustomReActParser(ReActSingleInputOutputParser):
    """Custom parser that handles JSON string inputs"""
    
    def parse(self, text: str):
        # First try the standard parser
        try:
            return super().parse(text)
        except Exception:
            pass
        
        # If that fails, try to extract and parse JSON from Action Input
        action_match = re.search(r"Action:\s*(.*?)[\n]", text)
        action_input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input_str = action_input_match.group(1).strip()
            
            # Try to parse the action input as JSON
            try:
                # Handle both JSON strings and already parsed objects
                if isinstance(action_input_str, str):
                    # Remove quotes if the whole thing is quoted
                    if action_input_str.startswith('"') and action_input_str.endswith('"'):
                        action_input_str = action_input_str[1:-1]
                    action_input = json_module.loads(action_input_str)
                else:
                    action_input = action_input_str
                    
                return AgentAction(tool=action, tool_input=action_input, log=text)
            except:
                # If JSON parsing fails, try to eval it (for simple dicts)
                try:
                    import ast
                    action_input = ast.literal_eval(action_input_str)
                    return AgentAction(tool=action, tool_input=action_input, log=text)
                except:
                    # Last resort - return as string
                    return AgentAction(tool=action, tool_input=action_input_str, log=text)
        
        # Check for Final Answer
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:\n|$)", text, re.DOTALL)
        if final_answer_match:
            return AgentFinish({"output": final_answer_match.group(1).strip()}, log=text)
        
        # If nothing matches, return the original parse error
        return super().parse(text)

async def main():
    print("=== PostgreSQL MCP Demo ===\n")
    
    # 1. Connect to MCP server
    print("1. Connecting to MCP server...")
    mcp_client = MultiServerMCPClient({
        "postgres": {
            "transport": "sse",
            "url": "http://localhost:8000/sse"
        }
    })
    
    # 2. Get available tools
    raw_tools = await mcp_client.get_tools()
    print(f"\n2. Available tools ({len(raw_tools)}):")
    for i, tool in enumerate(raw_tools):
        print(f"   {i+1}. {tool.name}: {tool.description}")
    
    # Wrap tools to handle string inputs
    from langchain_core.tools import StructuredTool
    tools = []
    for raw_tool in raw_tools:
        # Create a closure that captures the specific tool
        def make_wrapper(tool):
            async def tool_wrapper(input_data):
                # If input is a string, try to parse it as JSON
                if isinstance(input_data, str):
                    try:
                        input_data = json.loads(input_data)
                    except:
                        # If it's not JSON, create a dict with the expected key
                        if tool.name == "execute_sql":
                            input_data = {"sql": input_data}
                        else:
                            input_data = {"input": input_data}
                return await tool.ainvoke(input_data)
            return tool_wrapper
        
        # Create a wrapped tool with the same schema but flexible input handling
        wrapped_tool = StructuredTool(
            name=raw_tool.name,
            description=raw_tool.description,
            coroutine=make_wrapper(raw_tool),
            args_schema=None  # This allows string inputs
        )
        tools.append(wrapped_tool)
    
    # 3. Create LLM with tools (using your actual config)
    print("\n3. Creating LLM with tools...")
    from config import config
    
    print(f"   Model: {config.model_name}")
    print(f"   Base URL: {config.openai_base_url}")
    
    llm = ChatOpenAI(
        model=config.model_name,
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
        temperature=0
    )
    
    # 4. Create ReAct prompt with explicit tool instructions
    print("\n4. Creating ReAct prompt with tool instructions...")
    
    # Create the ReAct prompt template as specified in the documentation
    template = '''You are a helpful assistant with access to a PostgreSQL database.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action as a JSON object. For example: {{"sql": "SELECT * FROM users"}} or {{}} for no parameters
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For SQL queries, use the execute_sql tool. For example:
Action: execute_sql  
Action Input: {{"sql": "SELECT * FROM users WHERE age > 25"}}

Begin!

Question: {input}
Thought: {agent_scratchpad}'''
    
    prompt = PromptTemplate.from_template(template)
    print("   ✓ ReAct prompt template created")
    
    # 5. Create the ReAct agent with the prompt and custom parser
    print("\n5. Creating ReAct agent...")
    output_parser = CustomReActParser()
    agent = create_react_agent(llm, tools, prompt, output_parser=output_parser)
    
    # Create AgentExecutor to actually run the agent
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True
    )
    print("   ✓ Agent executor created")
    
    # 6. Now send query
    query = "Show me all users over 25 years old"
    print(f"\n6. Sending query to agent: '{query}'")
    
    try:
        # Agent executor uses 'input' key, not 'messages'
        result = await agent_executor.ainvoke({"input": query})
        
        print("\n7. Agent execution complete!")
        print("   Final output:", result.get("output", "No output"))
                
    except Exception as e:
        print(f"\n7. Agent execution failed: {type(e).__name__}: {str(e)}")
        
        # Let's test if our parser works by calling the tool directly with parsed JSON
        print("\n   Testing: Parsing and executing manually...")
        try:
            # Parse the action input manually
            action_input_str = '{"sql": "SELECT * FROM users WHERE age > 25"}'
            action_input = json.loads(action_input_str)
            
            # Find execute_sql tool
            execute_sql = next((t for t in tools if t.name == "execute_sql"), None)
            if execute_sql:
                result = await execute_sql.ainvoke(action_input)
                print("   Manual execution worked!")
                print(f"   Result: {json.dumps(result, indent=2)[:200]}...")
        except Exception as e2:
            print(f"   Manual execution also failed: {e2}")
    

async def interactive_mode():
    """Interactive mode for continuous conversation with the agent"""
    print("\n=== Interactive PostgreSQL Agent ===")
    print("Type 'exit' or 'quit' to stop, 'help' for available commands\n")
    
    # Setup MCP and agent (same as main())
    mcp_client = MultiServerMCPClient({
        "postgres": {
            "transport": "sse",
            "url": "http://localhost:8000/sse"
        }
    })
    
    raw_tools = await mcp_client.get_tools()
    print(f"\nConnected to MCP server with {len(raw_tools)} tools available:")
    for i, tool in enumerate(raw_tools, 1):
        print(f"  {i}. {tool.name}: {tool.description[:80]}...")
    
    # Wrap tools (same as in demo)
    from langchain_core.tools import StructuredTool
    tools = []
    for raw_tool in raw_tools:
        def make_wrapper(tool):
            async def tool_wrapper(input_data):
                if isinstance(input_data, str):
                    try:
                        input_data = json.loads(input_data)
                    except:
                        if tool.name == "execute_sql":
                            input_data = {"sql": input_data}
                        else:
                            input_data = {"input": input_data}
                return await tool.ainvoke(input_data)
            return tool_wrapper
        
        wrapped_tool = StructuredTool(
            name=raw_tool.name,
            description=raw_tool.description,
            coroutine=make_wrapper(raw_tool),
            args_schema=None
        )
        tools.append(wrapped_tool)
    
    from config import config
    llm = ChatOpenAI(
        model=config.model_name,
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
        temperature=0
    )
    
    # Create ReAct prompt
    template = '''You are a helpful assistant with access to a PostgreSQL database.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action as a JSON object. For example: {{"sql": "SELECT * FROM users"}} or {{}} for no parameters
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For SQL queries, use the execute_sql tool. For example:
Action: execute_sql  
Action Input: {{"sql": "SELECT * FROM users WHERE age > 25"}}

Begin!

Question: {input}
Thought: {agent_scratchpad}'''
    
    prompt = PromptTemplate.from_template(template)
    output_parser = CustomReActParser()
    agent = create_react_agent(llm, tools, prompt, output_parser=output_parser)
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    print("\nAgent ready! Ask questions about the database.\n")
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- Ask any question about the database")
                print("- 'tools' - List available tools")
                print("- 'exit' or 'quit' - Exit the program")
                continue
            
            if user_input.lower() == 'tools':
                print("\nAvailable tools:")
                for i, tool in enumerate(tools, 1):
                    print(f"{i}. {tool.name}: {tool.description}")
                continue
            
            if not user_input:
                continue
            
            print("\nAgent: Thinking...")
            
            try:
                result = await agent_executor.ainvoke({"input": user_input})
                print(f"\nAgent: {result.get('output', 'No response')}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
            continue

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())
