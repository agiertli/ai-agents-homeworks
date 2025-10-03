"""
Configuration management for the ReAct AI Agent
"""
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

class AgentConfig(BaseSettings):
    """Configuration settings for the AI Agent"""
    
    # OpenAI Compatible API Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field("http://localhost:11434/v1", env="OPENAI_BASE_URL")
    model_name: str = Field("llama2", env="MODEL_NAME")
    
    # PostgreSQL Configuration
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_db: str = Field("testdb", env="POSTGRES_DB")
    postgres_user: str = Field("testuser", env="POSTGRES_USER")
    postgres_password: str = Field("testpass", env="POSTGRES_PASSWORD")
    
    # MCP Server Configuration
    mcp_server_host: str = Field("localhost", env="MCP_SERVER_HOST")
    mcp_server_port: int = Field(8000, env="MCP_SERVER_PORT")
    
    # Agent Configuration
    agent_name: str = Field("PostgreSQL_ReAct_Agent", env="AGENT_NAME")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def mcp_server_url(self) -> str:
        """Generate MCP server URL"""
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}"

# Global configuration instance
config = AgentConfig()

