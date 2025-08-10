"""
Energo - Super Agent Framework
A powerful, enthusiastic AI agent with unlimited capabilities
"""

__version__ = "1.0.0"
__author__ = "Energo Team"
__description__ = "Super Agent with MCP integration, unlimited tools, and multi-AI provider support"

from .core import EnergoAgent
from .config import EnergoConfig
from .providers import AIProviderManager
from .tools import ToolManager
from .mcp import MCPConnector

__all__ = [
    "EnergoAgent",
    "EnergoConfig", 
    "AIProviderManager",
    "ToolManager",
    "MCPConnector"
]